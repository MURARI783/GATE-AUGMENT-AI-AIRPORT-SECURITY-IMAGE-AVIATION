from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from modules.augmentation import TECHNIQUES, apply_augmentations, create_demo_image
from modules.classifier import train_and_evaluate
from modules.generative_ai import generate_with_diffusion
from modules.preprocessing import decode_image, image_metadata, validate_upload

ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"

st.set_page_config(
    page_title="GateAugment AI | Security Imaging Lab",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy:#071522; --panel:#0d2030; --cyan:#35d5e8; --muted:#9bb0c2; }
    .stApp { background:linear-gradient(145deg,#071522 0%,#091b2b 55%,#0b2233 100%); color:#e7f2f8; }
    [data-testid="stSidebar"] { background:#081725; border-right:1px solid #16374a; }
    [data-testid="stMetric"] { background:#0d2030; border:1px solid #1b4052; padding:16px; border-radius:12px; }
    [data-testid="stMetricValue"] { color:#35d5e8; }
    h1,h2,h3 { color:#ecf8ff; }
    .eyebrow { color:#35d5e8; text-transform:uppercase; letter-spacing:.14em; font-size:.78rem; font-weight:700; }
    .hint { color:#9bb0c2; }
    .panel { background:#0d2030; border:1px solid #1b4052; border-radius:14px; padding:18px; }
    div.stButton > button[kind="primary"] { background:#12a9bd; border:0; color:#04131f; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "uploads": [],
        "generated": [],
        "activities": [],
        "training_result": None,
        "evaluation": None,
        "active_image": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def record_activity(message: str) -> None:
    st.session_state.activities.insert(0, {
        "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "event": message,
    })
    st.session_state.activities = st.session_state.activities[:12]


def upload_panel() -> None:
    files = st.file_uploader(
        "Choose X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload image files you have permission to use. Original files are not changed.",
    )
    if not files:
        return
    for uploaded in files:
        signature = f"{uploaded.name}:{uploaded.size}"
        if any(item["signature"] == signature for item in st.session_state.uploads):
            continue
        try:
            validate_upload(uploaded.name, uploaded.getvalue())
            image = decode_image(uploaded.getvalue())
            info = image_metadata(uploaded.name, uploaded.size, image)
            info["signature"] = signature
            info["image"] = image
            st.session_state.uploads.append(info)
            st.session_state.active_image = signature
            record_activity(f"Uploaded {uploaded.name}")
        except ValueError as exc:
            st.error(f"{uploaded.name}: {exc}")


def image_selector() -> dict | None:
    images = st.session_state.uploads
    if not images:
        st.info("Upload an image first, or use the labeled illustrative demo image.")
        if st.button("Load illustrative demo image"):
            demo = create_demo_image()
            info = image_metadata("illustrative_demo.png", 0, demo)
            info.update({"signature": "illustrative_demo", "image": demo, "demo": True})
            st.session_state.uploads.append(info)
            st.session_state.active_image = info["signature"]
            record_activity("Loaded illustrative demo image")
            st.rerun()
        return None
    labels = [item["name"] for item in images]
    selected_index = st.selectbox("Active image", range(len(labels)), format_func=lambda i: labels[i])
    return images[selected_index]


def show_uploads() -> None:
    st.markdown('<p class="eyebrow">Image intake</p>', unsafe_allow_html=True)
    st.title("Upload & inspect")
    st.caption("Supported formats: JPG, JPEG, PNG. Images remain local to this application.")
    upload_panel()
    if not st.session_state.uploads:
        return
    cols = st.columns(2)
    for index, info in enumerate(st.session_state.uploads):
        with cols[index % 2]:
            st.image(info["image"], caption=info["name"], use_container_width=True)
            st.caption(
                f"{info['width']} × {info['height']} px · {info['size_bytes']:,} bytes"
                + (" · illustrative demo, not a dataset image" if info.get("demo") else "")
            )


def save_generated(image, name: str, kind: str, techniques: list[str],
                   source_signature: str) -> dict:
    GENERATED_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = GENERATED_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    from PIL import Image

    Image.fromarray(image).save(path)
    entry = {
        "name": path.name,
        "path": str(path),
        "kind": kind,
        "techniques": techniques,
        "source_signature": source_signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": image,
    }
    st.session_state.generated.append(entry)
    return entry


def show_augmentation() -> None:
    st.markdown('<p class="eyebrow">Traditional image transforms</p>', unsafe_allow_html=True)
    st.title("Image augmentation")
    info = image_selector()
    if info is None:
        return
    if info.get("demo"):
        st.warning("Illustrative demo only. It is not a real baggage X-ray or research dataset sample.")
    selected = st.multiselect(
        "Select transformations",
        list(TECHNIQUES),
        default=["Rotation", "Horizontal flip", "Brightness / contrast"],
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        angle = st.slider("Rotation range (degrees)", 0, 90, 15)
        brightness = st.slider("Brightness", -0.5, 0.5, 0.15, 0.05)
    with c2:
        contrast = st.slider("Contrast", -0.5, 0.5, 0.15, 0.05)
        scale = st.slider("Scale variation", 0.7, 1.3, 1.1, 0.05)
    with c3:
        translate = st.slider("Translation (%)", 0, 20, 8)
        count = st.number_input("Variations per selected technique", 1, 12, 3)
    overlap_image = None
    if "Overlap composition" in selected:
        candidates = [i for i in st.session_state.uploads if i["signature"] != info["signature"]]
        if candidates:
            overlay = st.selectbox("Overlay image", candidates, format_func=lambda i: i["name"])
            overlap_image = overlay["image"]
        else:
            st.warning("Overlap composition needs a second uploaded image.")
    if st.button("Generate augmentations", type="primary", disabled=not selected):
        try:
            results = apply_augmentations(
                info["image"], selected, count=int(count), angle=angle,
                brightness=brightness, contrast=contrast, scale=scale,
                translate_percent=translate, overlay=overlap_image,
            )
            st.session_state.last_augmented = []
            for technique, image in results:
                entry = save_generated(
                    image, info["name"], "traditional", [technique], info["signature"],
                )
                st.session_state.last_augmented.append(entry)
            record_activity(f"Generated {len(results)} traditional augmentation(s)")
            st.success(f"Created {len(results)} transformed image(s).")
        except (ValueError, RuntimeError) as exc:
            st.error(f"Augmentation failed: {exc}")
    display = [
        item for item in getattr(st.session_state, "last_augmented", [])
        if item["source_signature"] == info["signature"]
    ]
    if display:
        columns = st.columns(min(4, len(display) + 1))
        with columns[0]:
            st.image(info["image"], caption="Original", use_container_width=True)
        for index, item in enumerate(display, start=1):
            with columns[index % len(columns)]:
                st.image(item["image"], caption=item["techniques"][0], use_container_width=True)
                st.download_button(
                    "Download PNG", data=Path(item["path"]).read_bytes(),
                    file_name=item["name"], mime="image/png", key=f"aug_dl_{item['name']}",
                )


def show_generative_studio() -> None:
    st.markdown('<p class="eyebrow">Model-backed synthesis</p>', unsafe_allow_html=True)
    st.title("Generative AI studio")
    st.warning(
        "The bundled diffusion integration uses a user-selected pretrained image-to-image model. "
        "Generic pretrained models are not trained or validated for X-ray imagery and may alter "
        "security-relevant structures; treat outputs as research-only until independently reviewed."
    )
    info = image_selector()
    if info is None:
        return
    model_id = st.text_input("Hugging Face Diffusers model ID", "stabilityai/stable-diffusion-2-1")
    strength = st.slider("Image-to-image strength", 0.1, 0.8, 0.25, 0.05)
    prompt = st.text_input("Prompt", "baggage x-ray scan, preserve the original object layout")
    fallback = st.checkbox("If model loading fails, use clearly labeled Albumentations fallback", value=True)
    if st.button("Generate with diffusion", type="primary"):
        try:
            generated = generate_with_diffusion(info["image"], model_id=model_id, prompt=prompt, strength=strength)
            entry = save_generated(
                generated, info["name"], "diffusion", [f"Diffusers: {model_id}"],
                info["signature"],
            )
            st.session_state.last_synthetic = entry
            record_activity(f"Generated image with diffusion model {model_id}")
        except (RuntimeError, ValueError, OSError) as exc:
            if fallback:
                try:
                    generated = apply_augmentations(info["image"], ["Brightness / contrast"], count=1)[0][1]
                    entry = save_generated(
                        generated, info["name"], "traditional_fallback",
                        ["Brightness / contrast"], info["signature"],
                    )
                    st.session_state.last_synthetic = entry
                    st.warning(f"Diffusion was unavailable ({exc}). Output below is a traditional Albumentations fallback, not generative synthesis.")
                    record_activity("Diffusion unavailable; used labeled traditional fallback")
                except (RuntimeError, ValueError) as fallback_error:
                    st.error(f"Diffusion failed ({exc}); fallback also failed: {fallback_error}")
            else:
                st.error(f"Diffusion model unavailable: {exc}")
    result = getattr(st.session_state, "last_synthetic", None)
    if result and result["source_signature"] == info["signature"]:
        left, right = st.columns(2)
        with left:
            st.image(info["image"], caption="Original input", use_container_width=True)
        with right:
            st.image(result["image"], caption=f"{result['kind'].replace('_', ' ').title()} · {result['name']}", use_container_width=True)
            st.download_button(
                "Download generated image", Path(result["path"]).read_bytes(),
                file_name=result["name"], mime="image/png",
            )


def show_training() -> None:
    st.markdown('<p class="eyebrow">TensorFlow model lab</p>', unsafe_allow_html=True)
    st.title("Train & compare classifiers")
    st.caption("Expected layout: dataset/train/<class_name>/* and dataset/test/<class_name>/*. Test data stays untouched; only training samples are augmented.")
    st.code(f"Training: {TRAIN_DIR}\nTesting:  {TEST_DIR}", language="text")
    epochs = st.slider("Epochs", 1, 30, 5)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
    if st.button("Train baseline and augmented models", type="primary"):
        with st.spinner("Training two models and evaluating against the same held-out test set…"):
            try:
                result = train_and_evaluate(
                    str(TRAIN_DIR), str(TEST_DIR), model_dir=str(MODEL_DIR),
                    epochs=epochs, batch_size=batch_size,
                )
                st.session_state.training_result = result
                st.session_state.evaluation = result["evaluation"]
                record_activity("Trained and evaluated baseline / augmented CNNs")
                st.success("Training and test-set evaluation completed.")
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                st.error(f"Training could not complete: {exc}")
    result = st.session_state.training_result
    if result:
        st.subheader("Actual held-out test results")
        cols = st.columns(4)
        for col, model_key in zip(cols, ["baseline", "augmented"]):
            metrics = result["evaluation"][model_key]
            col.metric(f"{model_key.title()} accuracy", f"{metrics['accuracy']:.3f}")
            col.caption(f"Precision {metrics['precision']:.3f} · Recall {metrics['recall']:.3f} · F1 {metrics['f1']:.3f}")
        for key in ["baseline", "augmented"]:
            history = result["history"][key]
            st.subheader(f"{key.title()} training curves")
            accuracy_chart, loss_chart = st.columns(2)
            with accuracy_chart:
                st.caption("Accuracy")
                st.line_chart({
                    "Training": history["accuracy"],
                    "Validation": history["val_accuracy"],
                })
            with loss_chart:
                st.caption("Loss")
                st.line_chart({
                    "Training": history["loss"],
                    "Validation": history["val_loss"],
                })


def show_evaluation() -> None:
    st.markdown('<p class="eyebrow">Metrics & diagnostics</p>', unsafe_allow_html=True)
    st.title("Model evaluation")
    result = st.session_state.training_result
    if not result:
        st.info("No evaluation results yet. Train the baseline and augmented models to populate this page.")
        return
    metrics = result["evaluation"]
    labels = result["class_names"]
    for model_key in ["baseline", "augmented"]:
        st.subheader(f"{model_key.title()} model")
        score = metrics[model_key]
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{score['accuracy']:.3f}")
        b.metric("Precision (macro)", f"{score['precision']:.3f}")
        c.metric("Recall (macro)", f"{score['recall']:.3f}")
        d.metric("F1 (macro)", f"{score['f1']:.3f}")
        matrix = score["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.8), max(3, len(labels) * 0.65)))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        for row in range(len(labels)):
            for col in range(len(labels)):
                ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def show_comparison() -> None:
    st.markdown('<p class="eyebrow">Visual review</p>', unsafe_allow_html=True)
    st.title("Results comparison")
    info = image_selector()
    if info is None:
        return
    items = [
        item for item in st.session_state.generated
        if item.get("source_signature") == info["signature"] and Path(item["path"]).exists()
    ]
    if not items:
        st.info("Generate transformed or diffusion images to compare them here.")
        return
    columns = st.columns(3)
    with columns[0]:
        st.image(info["image"], caption="Original", use_container_width=True)
    for index, item in enumerate(items):
        with columns[(index + 1) % len(columns)]:
            st.image(item["image"], caption=f"{item['kind'].replace('_', ' ').title()}: {', '.join(item['techniques'])}", use_container_width=True)
            st.download_button("Download", Path(item["path"]).read_bytes(), item["name"], "image/png", key=f"cmp_{item['name']}")


def build_report() -> bytes:
    training = st.session_state.training_result
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_images": [
            {key: value for key, value in item.items() if key not in {"image", "signature"}}
            for item in st.session_state.uploads
        ],
        "total_uploaded_images": sum(not item.get("demo", False) for item in st.session_state.uploads),
        "total_augmented_images": len(st.session_state.generated),
        "generated_images": [
            {key: value for key, value in item.items() if key != "image"}
            for item in st.session_state.generated
        ],
        "augmentation_techniques": sorted({
            technique for item in st.session_state.generated for technique in item["techniques"]
        }),
        "model_evaluation": training["evaluation"] if training else None,
        "training_history": training["history"] if training else None,
        "model_comparison": (
            {key: training["evaluation"][key]["accuracy"] for key in ["baseline", "augmented"]}
            if training else None
        ),
        "activities": st.session_state.activities,
        "dataset_note": "Metrics are omitted until actual models have been trained and evaluated on the configured test set.",
    }
    return json.dumps(report, indent=2, allow_nan=False).encode("utf-8")


def show_reports() -> None:
    st.markdown('<p class="eyebrow">Export & audit trail</p>', unsafe_allow_html=True)
    st.title("Reports & downloads")
    st.write("Export a JSON report containing upload metadata, generated-image provenance, selected techniques, activity history, and actual model metrics when available.")
    report_bytes = build_report()
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "gateaugment_report.json"
    report_path.write_bytes(report_bytes)
    st.download_button(
        "Download project report (JSON)", report_bytes,
        file_name="gateaugment_report.json", mime="application/json",
        type="primary",
    )
    items = [item for item in st.session_state.generated if Path(item["path"]).exists()]
    if items:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for item in items:
                bundle.write(item["path"], arcname=item["name"])
        st.download_button("Download all generated images (ZIP)", archive.getvalue(), "gateaugment_images.zip", "application/zip")
    if st.session_state.training_result:
        st.caption("Metrics in the report are computed from the current run's untouched test set.")
    else:
        st.info("No model metrics are reported because no model evaluation has been run.")


def show_dashboard() -> None:
    st.markdown('<p class="eyebrow">Security imaging research workspace</p>', unsafe_allow_html=True)
    st.title("GateAugment AI")
    st.caption("Baggage X-ray augmentation and classifier robustness workbench")
    training = st.session_state.training_result
    metrics = training["evaluation"] if training else None
    cols = st.columns(4)
    uploaded_count = sum(not item.get("demo", False) for item in st.session_state.uploads)
    cols[0].metric("Images uploaded", uploaded_count)
    cols[1].metric("Augmentations generated", len(st.session_state.generated))
    cols[2].metric("Baseline accuracy", f"{metrics['baseline']['accuracy']:.3f}" if metrics else "Not evaluated")
    cols[3].metric("Augmented accuracy", f"{metrics['augmented']['accuracy']:.3f}" if metrics else "Not evaluated")
    if not metrics:
        st.info("Model accuracy is shown only after a real training and held-out test evaluation.")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Workflow")
        st.markdown(
            "1. Upload authorized images or inspect the labeled demo image.  \n"
            "2. Apply configurable traditional transforms or attempt a pretrained diffusion model.  \n"
            "3. Organize a labeled dataset, then compare CNNs on the same untouched test set.  \n"
            "4. Review outputs and export a provenance report."
        )
    with right:
        st.subheader("Recent activity")
        if st.session_state.activities:
            for item in st.session_state.activities[:6]:
                st.write(f"`{item['time']}` · {item['event']}")
        else:
            st.caption("No activity yet.")
    st.warning("Research tool only. Validate every generated image and annotation; never use unreviewed synthetic imagery for operational security decisions.")


PAGES = {
    "Home dashboard": show_dashboard,
    "Image upload": show_uploads,
    "Image augmentation": show_augmentation,
    "Generative AI studio": show_generative_studio,
    "Model training": show_training,
    "Model evaluation": show_evaluation,
    "Results comparison": show_comparison,
    "Reports & downloads": show_reports,
}

initialize_state()
with st.sidebar:
    st.markdown("## ✈ GateAugment AI")
    st.caption("AIRPORT SECURITY · IMAGING LAB")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Local-first research application")

PAGES[page]()
