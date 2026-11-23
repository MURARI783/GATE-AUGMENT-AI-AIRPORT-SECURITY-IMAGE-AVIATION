from __future__ import annotations

import io
import json
import zipfile
import csv
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from modules.augmentation import (
    TECHNIQUES,
    apply_augmentations,
    apply_augmentations_with_boxes,
    create_demo_image,
)
from modules.classifier import train_and_evaluate
from modules.generative_ai import diffusion_availability, generate_with_diffusion
from modules.preprocessing import (
    decode_image,
    image_metadata,
    parse_yolo_annotations,
    validate_upload,
)
from modules.storage import (
    ensure_project_dirs,
    load_project_state,
    load_saved_items,
    save_project_state,
    save_uploaded_annotation,
    save_uploaded_image,
)

ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
DATASET_ROOT = ROOT / "dataset"

ensure_project_dirs(ROOT)

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
    if "storage_initialized" not in st.session_state:
        try:
            saved = load_project_state(ROOT)
            st.session_state.uploads = load_saved_items(ROOT, saved["uploads"], is_upload=True)
            st.session_state.generated = load_saved_items(ROOT, saved["generated"])
        except (RuntimeError, OSError) as exc:
            st.error(f"Could not restore saved project data: {exc}")
            st.stop()
        st.session_state.activities = saved["activities"]
        st.session_state.experiments = saved.get("experiments", [])
        evaluation_path = MODEL_DIR / "latest_evaluation.json"
        if evaluation_path.is_file():
            try:
                result = json.loads(evaluation_path.read_text(encoding="utf-8"))
                if all((MODEL_DIR / f"{name}_cnn.keras").is_file() for name in ("baseline", "augmented")):
                    st.session_state.training_result = result
                    st.session_state.evaluation = result.get("evaluation")
            except (OSError, json.JSONDecodeError) as exc:
                st.error(f"Could not restore saved evaluation: {exc}")
        st.session_state.storage_initialized = True
    defaults = {
        "uploads": [],
        "generated": [],
        "activities": [],
        "experiments": [],
        "training_result": None,
        "evaluation": None,
        "active_image": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def persist_state() -> None:
    try:
        save_project_state(ROOT, {
            "uploads": [
                {key: value for key, value in item.items() if key != "image"}
                for item in st.session_state.uploads
            ],
            "generated": [
                {key: value for key, value in item.items() if key != "image"}
                for item in st.session_state.generated
            ],
            "activities": st.session_state.activities,
            "experiments": st.session_state.experiments,
        })
    except OSError as exc:
        st.error(f"Could not save project history: {exc}")


def record_activity(message: str) -> None:
    st.session_state.activities.insert(0, {
        "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "event": message,
    })
    st.session_state.activities = st.session_state.activities[:12]
    persist_state()


def upload_panel() -> None:
    files = st.file_uploader(
        "Choose X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload image files you have permission to use. Original files are not changed.",
    )
    annotation_files = st.file_uploader(
        "Optional YOLO annotations (.txt), matched by image filename stem",
        type=["txt"],
        accept_multiple_files=True,
        key="yolo_annotation_uploader",
        help="Each row: class_id x_center y_center width height, normalized from 0 to 1.",
    )
    annotations_by_stem = {}
    try:
        annotations_by_stem = {
            Path(annotation.name).stem.casefold(): annotation.getvalue().decode("utf-8")
            for annotation in annotation_files
        }
    except UnicodeDecodeError as exc:
        st.error(f"Annotation files must be UTF-8 text: {exc}")
        return
    for uploaded in files or []:
        data = uploaded.getvalue()
        try:
            validate_upload(uploaded.name, data)
            image = decode_image(data)
            saved = save_uploaded_image(ROOT, uploaded.name, data)
            signature = saved["signature"]
            if any(item["signature"] == signature for item in st.session_state.uploads):
                continue
            info = image_metadata(uploaded.name, uploaded.size, image)
            info.update(saved)
            info["format"] = Path(uploaded.name).suffix[1:].upper()
            info["bboxes"], info["bbox_labels"] = [], []
            info["annotations_present"] = False
            info["image"] = image
            st.session_state.uploads.append(info)
            st.session_state.active_image = signature
            record_activity(f"Uploaded {uploaded.name}")
        except (ValueError, UnicodeDecodeError) as exc:
            st.error(f"{uploaded.name}: {exc}")
        except OSError as exc:
            st.error(f"Could not save {uploaded.name}: {exc}")
    for stem, label_text in annotations_by_stem.items():
        matches = [
            item for item in st.session_state.uploads
            if Path(item["name"]).stem.casefold() == stem
        ]
        if len(matches) != 1:
            st.error(
                f"Annotation '{stem}.txt' must match exactly one uploaded image; "
                f"found {len(matches)}."
            )
            continue
        try:
            boxes, labels = parse_yolo_annotations(label_text)
            info = matches[0]
            annotation_path = save_uploaded_annotation(ROOT, info["signature"], label_text)
            info.update({
                "bboxes": boxes,
                "bbox_labels": labels,
                "annotations_present": True,
                "annotation_path": annotation_path,
            })
            persist_state()
            record_activity(f"Added {len(boxes)} YOLO box(es) to {info['name']}")
        except ValueError as exc:
            st.error(f"{stem}.txt: {exc}")


def image_selector() -> dict | None:
    images = st.session_state.uploads
    if not images:
        st.info("Upload an image first, or use the labeled illustrative demo image.")
        if st.button("Load illustrative demo image"):
            demo = create_demo_image()
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(demo).save(buffer, format="PNG")
            saved = save_uploaded_image(ROOT, "illustrative_demo.png", buffer.getvalue())
            info = image_metadata("illustrative_demo.png", len(buffer.getvalue()), demo)
            info.update({
                **saved, "format": "PNG", "bboxes": [], "bbox_labels": [],
                "annotations_present": False, "image": demo, "demo": True,
            })
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
            annotation_description = (
                f" · {len(info.get('bboxes', []))} YOLO object box(es)"
                if info.get("annotations_present") else " · image-level labels only"
            )
            st.caption(
                f"{info['width']} × {info['height']} px · {info['size_bytes']:,} bytes · "
                f"{info.get('format', 'PNG')}{annotation_description}"
                + (" · illustrative demo, not a dataset image" if info.get("demo") else "")
            )


def save_generated(image, name: str, kind: str, techniques: list[str],
                   source_signature: str, bboxes=None, bbox_labels=None) -> dict:
    GENERATED_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = GENERATED_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    from PIL import Image

    Image.fromarray(image).save(path)
    if bboxes is not None and bbox_labels is not None:
        annotation_path = path.with_suffix(".txt")
        annotation_path.write_text(
            "".join(
                f"{label} {box[0]:.8f} {box[1]:.8f} {box[2]:.8f} {box[3]:.8f}\n"
                for box, label in zip(bboxes, bbox_labels)
            ),
            encoding="utf-8",
        )
    entry = {
        "name": path.name,
        "path": str(path),
        "kind": kind,
        "techniques": techniques,
        "source_signature": source_signature,
        "bboxes": bboxes,
        "bbox_labels": bbox_labels,
        "annotations_present": bboxes is not None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": image,
    }
    st.session_state.generated.append(entry)
    persist_state()
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
    overlay_info = None
    if "Overlap composition" in selected:
        candidates = [i for i in st.session_state.uploads if i["signature"] != info["signature"]]
        if candidates:
            overlay_info = st.selectbox("Overlay image", candidates, format_func=lambda i: i["name"])
            overlap_image = overlay_info["image"]
        else:
            st.warning("Overlap composition needs a second uploaded image.")
    if st.button("Generate augmentations", type="primary", disabled=not selected):
        try:
            has_annotations = info.get("annotations_present", False)
            overlay_annotated = bool(
                overlay_info and overlay_info.get("annotations_present", False)
            )
            if "Overlap composition" in selected and (
                has_annotations != overlay_annotated
            ):
                raise ValueError(
                    "Overlap composition requires YOLO annotations for both images or neither image."
                )
            if has_annotations or overlay_annotated:
                results = apply_augmentations_with_boxes(
                    info["image"], selected, info.get("bboxes", []),
                    info.get("bbox_labels", []), count=int(count), angle=angle,
                    brightness=brightness, contrast=contrast, scale=scale,
                    translate_percent=translate, overlay=overlap_image,
                    overlay_bboxes=overlay_info.get("bboxes", []) if overlay_info else None,
                    overlay_bbox_labels=overlay_info.get("bbox_labels", []) if overlay_info else None,
                )
            else:
                results = apply_augmentations(
                    info["image"], selected, count=int(count), angle=angle,
                    brightness=brightness, contrast=contrast, scale=scale,
                    translate_percent=translate, overlay=overlap_image,
                )
            st.session_state.last_augmented = []
            for result_item in results:
                if len(result_item) == 4:
                    technique, image, boxes, labels = result_item
                else:
                    technique, image = result_item
                    boxes, labels = None, None
                entry = save_generated(
                    image, info["name"], "traditional", [technique],
                    info["signature"], boxes, labels,
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
                if item.get("annotations_present"):
                    annotation_path = Path(item["path"]).with_suffix(".txt")
                    if annotation_path.is_file():
                        st.download_button(
                            "Download YOLO boxes",
                            annotation_path.read_bytes(),
                            file_name=annotation_path.name,
                            mime="text/plain",
                            key=f"aug_label_{item['name']}",
                        )


def show_generative_studio() -> None:
    st.markdown('<p class="eyebrow">Model-backed synthesis</p>', unsafe_allow_html=True)
    st.title("Generative AI studio")
    st.warning(
        "The bundled diffusion integration uses a user-selected pretrained image-to-image model. "
        "Generic pretrained models are not trained or validated for X-ray imagery and may alter "
        "security-relevant structures; treat outputs as research-only until independently reviewed."
    )
    diffusion_ready, diffusion_message = diffusion_availability()
    if diffusion_ready:
        st.info(diffusion_message)
    else:
        st.error(diffusion_message)
        st.caption("Traditional Albumentations fallback remains available when enabled below.")
    info = image_selector()
    if info is None:
        return
    model_id = st.text_input("Hugging Face Diffusers model ID", "stabilityai/stable-diffusion-2-1")
    strength = st.slider("Image-to-image strength", 0.1, 0.8, 0.25, 0.05)
    prompt = st.text_input("Prompt", "baggage x-ray scan, preserve the original object layout")
    fallback = st.checkbox("If model loading fails, use clearly labeled Albumentations fallback", value=True)
    if st.button("Generate with diffusion", type="primary"):
        with st.spinner("Loading model weights and generating an image. First use may download a large model..."):
            try:
                generated = generate_with_diffusion(info["image"], model_id=model_id, prompt=prompt, strength=strength)
                entry = save_generated(
                    generated, info["name"], "diffusion", [f"Diffusers: {model_id}"],
                    info["signature"],
                )
                st.session_state.last_synthetic = entry
                if info.get("annotations_present"):
                    st.warning(
                        "Diffusion can change object geometry. Source YOLO boxes were not copied "
                        "to the synthetic image; annotate and validate it before training."
                    )
                record_activity(f"Generated image with diffusion model {model_id}")
            except (RuntimeError, ValueError, OSError) as exc:
                if fallback:
                    try:
                        generated = apply_augmentations(info["image"], ["Brightness / contrast"], count=1)[0][1]
                        fallback_boxes = info.get("bboxes") if info.get("annotations_present") else None
                        fallback_labels = info.get("bbox_labels") if info.get("annotations_present") else None
                        entry = save_generated(
                            generated, info["name"], "traditional_fallback",
                            ["Brightness / contrast"], info["signature"],
                            fallback_boxes, fallback_labels,
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


def discover_datasets() -> dict[str, tuple[Path, Path]]:
    candidates = {}
    if (DATASET_ROOT / "train").is_dir() and (DATASET_ROOT / "test").is_dir():
        candidates["Default dataset"] = (DATASET_ROOT / "train", DATASET_ROOT / "test")
    for folder in DATASET_ROOT.iterdir():
        if not folder.is_dir():
            continue
        train, test = folder / "train", folder / "test"
        if train.is_dir() and test.is_dir():
            candidates[folder.name] = (train, test)
    return candidates


def show_training() -> None:
    st.markdown('<p class="eyebrow">TensorFlow model lab</p>', unsafe_allow_html=True)
    st.title("Train & compare classifiers")
    st.caption("Select a prepared dataset. The training folder is split into training/validation; the separate test folder is untouched.")
    datasets = discover_datasets()
    if datasets:
        selected_dataset = st.selectbox("Dataset", list(datasets), key="selected_dataset")
        train_dir, test_dir = datasets[selected_dataset]
    else:
        train_dir, test_dir = TRAIN_DIR, TEST_DIR
        st.warning("No dataset with matching train/test folders was found. Arrange data using the README layout.")
    st.code(f"Training: {train_dir}\nTesting:  {test_dir}", language="text")
    epochs = st.slider("Epochs", 1, 30, 5)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
    validation_split = st.slider("Training/validation split", 0.1, 0.4, 0.2, 0.05)
    if st.button("Train baseline and augmented models", type="primary"):
        progress = st.progress(0, text="Preparing training data...")
        status = st.empty()

        def update_training_progress(fraction: float, message: str) -> None:
            progress.progress(max(0.0, min(1.0, fraction)), text=message)
            status.info(message)

        try:
            result = train_and_evaluate(
                str(train_dir), str(test_dir), model_dir=str(MODEL_DIR),
                epochs=epochs, batch_size=batch_size,
                validation_split=validation_split,
                progress_callback=update_training_progress,
            )
            st.session_state.training_result = result
            st.session_state.evaluation = result["evaluation"]
            st.session_state.experiments.append({
                "completed_at": result["completed_at"],
                "epochs": result["epochs"],
                "batch_size": result["batch_size"],
                "class_names": result["class_names"],
                "evaluation": result["evaluation"],
                "train_samples": result["train_samples"],
                "validation_samples": result["validation_samples"],
                "validation_split": result["validation_split"],
                "test_set": result["test_set"],
            })
            persist_state()
            record_activity("Trained and evaluated baseline / augmented CNNs")
            st.success("Training and test-set evaluation completed. Models and training history were saved.")
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            progress.empty()
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
        st.caption(
            f"Saved models: models/baseline_cnn.keras and models/augmented_cnn.keras · "
            f"classes: {', '.join(result['class_names'])} · "
            f"training/validation samples: {result.get('train_samples', 'n/a')}/"
            f"{result.get('validation_samples', 'n/a')}"
        )


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
        report = pd.DataFrame(score["classification_report"]).transpose()
        st.caption("Per-class classification report")
        st.dataframe(report, use_container_width=True)
        history = result["history"][model_key]
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"{model_key.title()} accuracy")
            st.line_chart({
                "Training": history["accuracy"],
                "Validation": history["val_accuracy"],
            })
        with c2:
            st.caption(f"{model_key.title()} loss")
            st.line_chart({
                "Training": history["loss"],
                "Validation": history["val_loss"],
            })


def show_comparison() -> None:
    st.markdown('<p class="eyebrow">Visual review</p>', unsafe_allow_html=True)
    st.title("Results comparison")
    if st.session_state.uploads:
        info = image_selector()
        items = [
            item for item in st.session_state.generated
            if item.get("source_signature") == info["signature"] and Path(item["path"]).exists()
        ]
        if items:
            columns = st.columns(min(4, len(items) + 1))
            with columns[0]:
                st.image(info["image"], caption="Original", use_container_width=True)
            for index, item in enumerate(items, start=1):
                with columns[index % len(columns)]:
                    st.image(item["image"], caption=f"{item['kind'].replace('_', ' ').title()}: {', '.join(item['techniques'])}", use_container_width=True)
                    st.download_button("Download", Path(item["path"]).read_bytes(), item["name"], "image/png", key=f"cmp_{item['name']}")
        else:
            st.info("No generated variation exists for this image yet.")
    else:
        st.info("Upload an image to compare original and augmented visuals.")

    result = st.session_state.training_result
    if not result:
        st.info("Model performance comparison appears after both CNNs have been evaluated.")
        return
    baseline = result["evaluation"]["baseline"]
    augmented = result["evaluation"]["augmented"]
    st.subheader("Held-out test performance")
    metric_rows = []
    for metric in ("accuracy", "precision", "recall", "f1"):
        metric_rows.append({
            "Metric": metric.title(),
            "Original-data model": baseline[metric],
            "Augmented-data model": augmented[metric],
            "Difference": augmented[metric] - baseline[metric],
        })
    st.dataframe(pd.DataFrame(metric_rows).set_index("Metric"), use_container_width=True)
    st.bar_chart(pd.DataFrame({
        "Original-data model": [baseline["accuracy"], baseline["precision"], baseline["recall"]],
        "Augmented-data model": [augmented["accuracy"], augmented["precision"], augmented["recall"]],
    }, index=["Accuracy", "Precision", "Recall"]))


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
        "trained_model_status": all(
            (MODEL_DIR / f"{name}_cnn.keras").is_file()
            for name in ("baseline", "augmented")
        ),
        "generated_images": [
            {key: value for key, value in item.items() if key != "image"}
            for item in st.session_state.generated
        ],
        "augmentation_techniques": sorted({
            technique for item in st.session_state.generated for technique in item["techniques"]
        }),
        "model_evaluation": training["evaluation"] if training else None,
        "training_history": training["history"] if training else None,
        "model_details": ({
            key: training.get(key) for key in (
                "model_architecture", "augmentation_method", "epochs",
                "batch_size", "image_size", "class_names", "test_set",
                "validation_split", "completed_at", "experiment_id",
            )
        } if training else None),
        "model_comparison": (
            {key: training["evaluation"][key]["accuracy"] for key in ["baseline", "augmented"]}
            if training else None
        ),
        "activities": st.session_state.activities,
        "experiments": st.session_state.experiments,
        "dataset_note": "Metrics are omitted until actual models have been trained and evaluated on the configured test set.",
    }
    return json.dumps(report, indent=2, allow_nan=False).encode("utf-8")


def build_csv_report() -> bytes:
    training = st.session_state.training_result
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Project", "GateAugment AI"])
    writer.writerow(["Experiment date (UTC)", datetime.now(timezone.utc).isoformat()])
    writer.writerow(["Uploaded images", sum(not item.get("demo", False) for item in st.session_state.uploads)])
    writer.writerow(["Augmented/generated images", len(st.session_state.generated)])
    techniques = sorted({
        technique for item in st.session_state.generated for technique in item["techniques"]
    })
    writer.writerow(["Techniques", "; ".join(techniques)])
    writer.writerow([])
    writer.writerow(["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"])
    if training:
        for model_key, metrics in training["evaluation"].items():
            writer.writerow([
                model_key, metrics["accuracy"], metrics["precision"],
                metrics["recall"], metrics["f1"],
            ])
        writer.writerow([])
        writer.writerow(["Model details", "Value"])
        writer.writerow(["Architecture", training.get("model_architecture", "CNN")])
        writer.writerow(["Epochs", training.get("epochs")])
        writer.writerow(["Batch size", training.get("batch_size")])
        writer.writerow(["Validation split", training.get("validation_split")])
        writer.writerow(["Classes", "; ".join(training.get("class_names", []))])
        writer.writerow(["Test dataset", training.get("test_set")])
    else:
        writer.writerow(["No evaluated model metrics", "Train and evaluate both models to populate metrics."])
    return output.getvalue().encode("utf-8-sig")


def show_reports() -> None:
    st.markdown('<p class="eyebrow">Export & audit trail</p>', unsafe_allow_html=True)
    st.title("Reports & downloads")
    st.write("Export CSV or JSON experiment reports with uploaded/generated image counts, provenance, model details, and actual metrics when available.")
    report_bytes = build_report()
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "gateaugment_report.json"
    report_path.write_bytes(report_bytes)
    csv_bytes = build_csv_report()
    (REPORT_DIR / "gateaugment_report.csv").write_bytes(csv_bytes)
    json_col, csv_col = st.columns(2)
    with json_col:
        st.download_button(
            "Download project report (JSON)", report_bytes,
            file_name="gateaugment_report.json", mime="application/json",
            type="primary",
        )
    with csv_col:
        st.download_button(
            "Download experiment summary (CSV)", csv_bytes,
            file_name="gateaugment_report.csv", mime="text/csv",
        )
    items = [item for item in st.session_state.generated if Path(item["path"]).exists()]
    if items:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for item in items:
                bundle.write(item["path"], arcname=item["name"])
                annotation_path = Path(item["path"]).with_suffix(".txt")
                if annotation_path.is_file():
                    bundle.write(annotation_path, arcname=annotation_path.name)
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
    trained_model_count = sum(
        (MODEL_DIR / f"{name}_cnn.keras").is_file()
        for name in ("baseline", "augmented")
    )
    cols = st.columns(4)
    uploaded_count = sum(not item.get("demo", False) for item in st.session_state.uploads)
    cols[0].metric("Images uploaded", uploaded_count)
    cols[1].metric("Augmentations generated", len(st.session_state.generated))
    cols[2].metric("Trained models", f"{trained_model_count} / 2")
    cols[3].metric("Test accuracy", f"{metrics['augmented']['accuracy']:.3f}" if metrics else "Not evaluated")
    if not metrics:
        st.info("Model accuracy is shown only after a real training and held-out test evaluation.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Image activity")
        timeline = []
        for item in st.session_state.uploads:
            if not item.get("demo", False) and item.get("uploaded_at"):
                timeline.append({
                    "Date": item["uploaded_at"][:10],
                    "Uploaded": 1,
                    "Generated": 0,
                })
        for item in st.session_state.generated:
            if item.get("created_at"):
                timeline.append({
                    "Date": item["created_at"][:10],
                    "Uploaded": 0,
                    "Generated": 1,
                })
        if timeline:
            activity_chart = pd.DataFrame(timeline).groupby("Date")[["Uploaded", "Generated"]].sum()
            st.area_chart(activity_chart, use_container_width=True)
        else:
            st.caption("Upload or generate images to populate activity charts.")
    with chart_right:
        st.subheader("CNN performance")
        if metrics:
            performance = pd.DataFrame({
                "Baseline": [
                    metrics["baseline"][key] for key in ("accuracy", "precision", "recall", "f1")
                ],
                "Augmented": [
                    metrics["augmented"][key] for key in ("accuracy", "precision", "recall", "f1")
                ],
            }, index=["Accuracy", "Precision", "Recall", "F1"])
            st.bar_chart(performance, use_container_width=True)
        else:
            st.caption("Training and evaluation metrics appear here after a real model run.")
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
    if st.session_state.experiments:
        st.subheader("Experiment history")
        history_rows = []
        for experiment in st.session_state.experiments:
            history_rows.append({
                "Completed (UTC)": experiment["completed_at"],
                "Baseline accuracy": experiment["evaluation"]["baseline"]["accuracy"],
                "Augmented accuracy": experiment["evaluation"]["augmented"]["accuracy"],
                "Classes": ", ".join(experiment["class_names"]),
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)


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
