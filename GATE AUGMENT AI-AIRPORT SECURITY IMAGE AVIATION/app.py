from __future__ import annotations

import io
import json
import zipfile
import csv
import uuid
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st

from modules.augmentation import (
    TECHNIQUES,
    apply_augmentations,
    apply_augmentations_with_boxes,
    create_demo_image,
)
from modules.classifier import train_and_evaluate
from modules.database import event_summary, log_event, recent_events, recent_scans, record_scan
from modules.detection import detection_availability, detect_objects
from modules.generative_ai import (
    diffusion_availability,
    generate_many_with_diffusion,
)
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
from modules.scanner import decode_video, process_scan, validate_processed_image
from modules.reports import build_pdf_report

ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
DATASET_ROOT = ROOT / "dataset"
DB_PATH = REPORT_DIR / "gateaugment.sqlite3"

ensure_project_dirs(ROOT)
event_summary(DB_PATH)

st.set_page_config(
    page_title="GateAugment AI | Security Imaging Lab",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy:#12304a; --panel:#ffffff; --cyan:#0891b2;
        --muted:#526b80; --blue:#2563eb; --mint:#0f9f83;
    }
    .stApp {
        background:linear-gradient(135deg,#eaf7ff 0%,#f6f1ff 52%,#eafff5 100%);
        color:#17324d;
    }
    [data-testid="stAppViewContainer"] { background:transparent; }
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#dff5ff 0%,#eefaff 52%,#e6fff6 100%);
        border-right:1px solid #b9ddeb;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label { color:#12304a; }
    [data-testid="stMetric"] {
        background:linear-gradient(145deg,#ffffff 0%,#f1fbff 100%);
        color:#17324d;
        border:1px solid #c5e2ef;
        border-left:4px solid #06a6c8;
        padding:16px;
        border-radius:14px;
        box-shadow:0 6px 20px rgba(27,82,117,.08);
    }
    [data-testid="stMetricValue"] { color:#087e8b; }
    [data-testid="stMetricLabel"] { color:#49657c; }
    .main h1,.main h2,.main h3 { color:#12304a; }
    .main p,.main label,.main [data-testid="stMarkdownContainer"] {
        color:#29485f;
    }
    .eyebrow { color:#07869a; text-transform:uppercase; letter-spacing:.14em; font-size:.78rem; font-weight:700; }
    .main .eyebrow { color:#07869a; }
    .hint { color:#526b80; }
    .panel {
        background:linear-gradient(145deg,#ffffff,#f0faff);
        border:1px solid #c5e2ef;
        border-radius:14px;
        padding:18px;
        box-shadow:0 6px 20px rgba(27,82,117,.06);
    }
    .main input,.main textarea,.main [data-baseweb="select"] > div {
        background-color:#ffffff;
        color:#17324d;
        border-color:#b9d4e5;
    }
    .main [data-testid="stAlert"] { border-radius:12px; }
    .main [data-testid="stDataFrame"],
    .main [data-testid="stVegaLiteChart"],
    .main [data-testid="stArrowVegaLiteChart"] {
        background:#ffffff;
        border:1px solid #d2e7f0;
        border-radius:14px;
        box-shadow:0 5px 18px rgba(17,74,103,.05);
    }
    div.stButton > button {
        background:#ffffff;
        border:1px solid #9bcfe0;
        color:#12304a;
        border-radius:10px;
    }
    div.stButton > button:hover {
        border-color:#078fae;
        color:#075c73;
        background:#e9faff;
    }
    div.stButton > button[kind="primary"] {
        background:linear-gradient(100deg,#0891b2,#2563eb);
        border:0;
        color:#ffffff;
        font-weight:700;
    }
    div.stDownloadButton > button {
        background:linear-gradient(100deg,#0f9f83,#0b8fa8);
        border:0;
        color:#ffffff;
        font-weight:700;
        border-radius:10px;
    }
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
        "scanner_frames": [],
        "scanner_frame_index": 0,
        "scanner_status": "idle",
        "scanner_results": [],
        "detection_result": None,
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
    try:
        log_event(DB_PATH, "activity", message)
    except (OSError, ValueError, sqlite3.Error):
        st.warning("Project activity was saved, but the analytics database could not be updated.")


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


def save_scan_result(
    image: np.ndarray, *, source: str, filename: str,
    operation: str, processing_ms: float, frame_index: int,
    status: str = "Processed",
) -> dict:
    scan_dir = GENERATED_DIR / "scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    scan_id = uuid.uuid4().hex[:12]
    path = scan_dir / f"scan_{scan_id}.png"
    from PIL import Image

    Image.fromarray(image).save(path)
    height, width = image.shape[:2]
    scan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "filename": filename,
        "status": status,
        "processing_ms": processing_ms,
        "width": width,
        "height": height,
        "operation": operation,
        "result_path": str(path),
        "metadata": {"frame_index": frame_index},
    }
    record_scan(DB_PATH, scan)
    scan["image"] = image
    st.session_state.scanner_results.insert(0, scan)
    return scan


def show_scanner() -> None:
    st.markdown('<p class="eyebrow">Scanner workstation · research/demo only</p>', unsafe_allow_html=True)
    st.title("Live scanner")
    st.error(
        "This app does not connect to airport X-ray hardware. Uploaded media is processed locally. "
        "Webcam mode captures an ordinary camera still only and is not an X-ray scanner."
    )
    mode = st.radio(
        "Input source",
        ["Uploaded image", "Uploaded video", "Webcam demonstration"],
        horizontal=True,
    )
    operation = st.selectbox(
        "Image enhancement",
        ["Enhance", "Grayscale", "Contrast enhancement", "Edge detection", "Sharpen"],
    )
    source_name = ""
    source_kind = "image"
    source_frames = []
    webcam_capture = None

    if mode == "Uploaded image":
        available = [item for item in st.session_state.uploads if not item.get("demo")]
        if not available:
            st.info("Upload image scans on the Image Upload page first.")
        else:
            selected_index = st.selectbox(
                "Image scan", range(len(available)),
                format_func=lambda index: available[index]["name"],
            )
            selected = available[selected_index]
            source_name = selected["name"]
            source_frames = [selected["image"]]
    elif mode == "Uploaded video":
        source_kind = "video"
        uploaded_video = st.file_uploader(
            "Upload video scan (MP4, MOV, AVI, MKV or WebM; max 100 MB)",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="scanner_video",
        )
        if uploaded_video is not None:
            source_name = uploaded_video.name
            video_signature = f"{uploaded_video.name}:{uploaded_video.size}"
            if st.button("Load video frames", type="primary"):
                try:
                    frames, fps = decode_video(
                        uploaded_video.getvalue(), Path(uploaded_video.name).suffix,
                    )
                    st.session_state.scanner_frames = frames
                    st.session_state.scanner_frame_index = 0
                    st.session_state.scanner_results = []
                    st.session_state.scanner_video_fps = fps
                    st.session_state.scanner_video_name = uploaded_video.name
                    st.session_state.scanner_loaded_source = video_signature
                    st.session_state.scanner_status = "ready"
                    st.success(
                        f"Loaded {len(frames)} sampled frames; source FPS {fps:.2f}. "
                        "Frames are sampled for review, not real-time security screening."
                    )
                except (ValueError, OSError) as exc:
                    st.error(f"Could not load video: {exc}")
            if st.session_state.get("scanner_loaded_source") == video_signature:
                source_frames = st.session_state.scanner_frames
            else:
                source_frames = []
        else:
            source_frames = []
        source_name = getattr(st.session_state, "scanner_video_name", source_name)
    else:
        webcam_capture = st.camera_input(
            "Capture a regular webcam still (demonstration only)",
            key="scanner_webcam_capture",
        )
        if webcam_capture is not None:
            source_name = "webcam_demo_capture.png"
            source_kind = "webcam_demo"
            try:
                source_frames = [decode_image(webcam_capture.getvalue())]
                st.session_state.scanner_frames = source_frames
                st.session_state.scanner_frame_index = 0
                st.session_state.scanner_results = []
                st.session_state.scanner_status = "ready"
            except ValueError as exc:
                st.error(f"Could not decode camera capture: {exc}")

    if mode == "Uploaded image" and source_frames:
        if st.session_state.get("scanner_loaded_source") != selected["signature"]:
            st.session_state.scanner_frames = source_frames
            st.session_state.scanner_frame_index = 0
            st.session_state.scanner_results = []
            st.session_state.scanner_status = "ready"
            st.session_state.scanner_loaded_source = selected["signature"]
        st.session_state.scanner_source_name = source_name
        st.session_state.scanner_source_kind = source_kind
    elif mode == "Webcam demonstration" and source_frames:
        camera_signature = hashlib.sha256(webcam_capture.getvalue()).hexdigest()
        if st.session_state.get("scanner_loaded_source") != camera_signature:
            st.session_state.scanner_loaded_source = camera_signature
            st.session_state.scanner_frames = source_frames
            st.session_state.scanner_frame_index = 0
            st.session_state.scanner_results = []
            st.session_state.scanner_status = "ready"
        st.session_state.scanner_source_name = source_name
        st.session_state.scanner_source_kind = source_kind
    elif mode == "Uploaded video" and st.session_state.scanner_frames:
        st.session_state.scanner_source_name = source_name
        st.session_state.scanner_source_kind = source_kind

    frames = source_frames
    status = st.session_state.scanner_status
    status_colors = {
        "idle": "🔵", "ready": "🟡", "running": "🟢", "paused": "🟠",
        "stopped": "⚪", "complete": "✅",
    }
    st.subheader(f"{status_colors.get(status, '⚪')} Scanner status: {status.title()}")
    if status == "running":
        st.markdown(
            """<style>
            .scan-track{height:5px;background:#103245;overflow:hidden;border-radius:4px;margin:12px 0}
            .scan-line{height:5px;width:25%;background:#35d5e8;box-shadow:0 0 14px #35d5e8;
            animation:scan 1.8s linear infinite}
            @keyframes scan{from{transform:translateX(-100%)}to{transform:translateX(500%)}}
            </style><div class="scan-track"><div class="scan-line"></div></div>""",
            unsafe_allow_html=True,
        )
    if frames:
        current_index = st.session_state.scanner_frame_index
        st.caption(
            f"Loaded frames: {len(frames)} · next frame: {min(current_index + 1, len(frames))}/"
            f"{len(frames)} · {source_name}"
        )
    controls = st.columns(4)
    if controls[0].button("Start scan", type="primary", disabled=not frames):
        if status in {"complete", "stopped"}:
            st.session_state.scanner_frame_index = 0
            st.session_state.scanner_results = []
        st.session_state.scanner_status = "running"
    if controls[1].button("Pause", disabled=status != "running"):
        st.session_state.scanner_status = "paused"
    if controls[2].button("Resume", disabled=status != "paused"):
        st.session_state.scanner_status = "running"
    if controls[3].button("Stop", disabled=status not in {"running", "paused"}):
        st.session_state.scanner_status = "stopped"

    if frames and status in {"ready", "paused", "stopped"}:
        if st.button("Capture current frame"):
            frame_index = min(st.session_state.scanner_frame_index, len(frames) - 1)
            frame = frames[frame_index]
            save_scan_result(
                frame,
                source=st.session_state.get("scanner_source_kind", "image"),
                filename=st.session_state.get("scanner_source_name", source_name),
                operation="Capture frame (unprocessed)",
                processing_ms=0.0,
                frame_index=frame_index,
                status="Captured",
            )
            st.success("Raw frame captured without enhancement.")

    batch_size = st.select_slider("Frames per processing step", options=[1, 5, 10], value=1)
    if st.button(
        "Process next frame batch",
        disabled=st.session_state.scanner_status != "running" or not frames,
    ):
        start_index = st.session_state.scanner_frame_index
        end_index = min(start_index + batch_size, len(frames))
        progress = st.progress(0, text="Processing frames…")
        for frame_index in range(start_index, end_index):
            try:
                processed, elapsed = process_scan(frames[frame_index], operation)
                validate_processed_image(processed, frames[frame_index].shape)
                result = save_scan_result(
                    processed,
                    source=st.session_state.get("scanner_source_kind", "image"),
                    filename=st.session_state.get("scanner_source_name", source_name),
                    operation=operation,
                    processing_ms=elapsed,
                    frame_index=frame_index,
                )
                progress.progress(
                    (frame_index - start_index + 1) / max(1, end_index - start_index),
                    text=f"Processed frame {frame_index + 1}",
                )
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(f"Frame {frame_index + 1} failed: {exc}")
                break
        st.session_state.scanner_frame_index = end_index
        if end_index >= len(frames):
            st.session_state.scanner_status = "complete"
        else:
            st.session_state.scanner_status = "paused"
        log_event(
            DB_PATH, "scan_batch",
            f"Processed frames {start_index + 1}–{end_index} from {source_name}",
            {"operation": operation, "source": st.session_state.get("scanner_source_kind", "image")},
        )

    results = st.session_state.scanner_results
    if results:
        latest = results[0]
        original_index = latest["metadata"]["frame_index"]
        original_frame = frames[min(original_index, len(frames) - 1)] if frames else None
        original_col, processed_col = st.columns(2)
        if original_frame is not None:
            original_col.image(original_frame, caption="Original input frame", use_container_width=True)
        processed_col.image(latest["image"], caption=f"Processed · {latest['operation']}", use_container_width=True)
        st.caption(
            f"{latest['width']} × {latest['height']} px · "
            f"{latest['processing_ms']:.2f} ms · {latest['status']} · "
            f"{latest['created_at']}"
        )
        st.download_button(
            "Download processed scan frame",
            Path(latest["result_path"]).read_bytes(),
            file_name=Path(latest["result_path"]).name,
            mime="image/png",
        )

    history = recent_scans(DB_PATH)
    if history:
        st.subheader("Recent scan history")
        st.dataframe(pd.DataFrame([
            {key: item[key] for key in (
                "created_at", "source", "filename", "operation",
                "status", "processing_ms", "width", "height",
            )}
            for item in history
        ]), use_container_width=True)


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
    try:
        log_event(
            DB_PATH,
            "synthetic" if kind == "diffusion" else "augmentation",
            f"Saved {kind} image {path.name}",
            {"techniques": techniques, "source_signature": source_signature},
        )
    except sqlite3.Error as exc:
        st.warning(f"Image was saved, but analytics history could not be updated: {exc}")
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
    batch_mode = st.checkbox("Apply selected techniques to all uploaded images", value=False)
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
    noise_sigma = st.slider("Gaussian noise σ", 0.0, 40.0, 8.0, 1.0)
    blur_kernel = st.select_slider("Blur kernel", options=[3, 5, 7, 9], value=3)
    overlap_image = None
    overlay_info = None
    if "Overlap composition" in selected:
        candidates = [i for i in st.session_state.uploads if i["signature"] != info["signature"]]
        if candidates:
            overlay_index = st.selectbox(
                "Overlay image", range(len(candidates)),
                format_func=lambda index: candidates[index]["name"],
            )
            overlay_info = candidates[overlay_index]
            overlap_image = overlay_info["image"]
        else:
            st.warning("Overlap composition needs a second uploaded image.")
    if st.button("Generate augmentations", type="primary", disabled=not selected):
        try:
            st.session_state.last_augmented = []
            sources = (
                [item for item in st.session_state.uploads if not item.get("demo")]
                if batch_mode else [info]
            )
            total_created = 0
            for source in sources:
                source_overlay = overlay_info
                if (
                    "Overlap composition" in selected
                    and source_overlay is not None
                    and source_overlay["signature"] == source["signature"]
                ):
                    paired = [
                        item for item in st.session_state.uploads
                        if item["signature"] != source["signature"]
                    ]
                    source_overlay = paired[0] if paired else None
                source_overlay_image = source_overlay["image"] if source_overlay else None
                has_annotations = source.get("annotations_present", False)
                overlay_annotated = bool(
                    source_overlay and source_overlay.get("annotations_present", False)
                )
                if "Overlap composition" in selected and (
                    has_annotations != overlay_annotated
                ):
                    raise ValueError(
                        "Overlap composition requires YOLO annotations for both images or neither image."
                    )
                augmentation_options = {
                    "count": int(count), "angle": angle, "brightness": brightness,
                    "contrast": contrast, "scale": scale,
                    "translate_percent": translate, "overlay": source_overlay_image,
                    "noise_sigma": noise_sigma, "blur_kernel": blur_kernel,
                }
                if has_annotations or overlay_annotated:
                    results = apply_augmentations_with_boxes(
                        source["image"], selected, source.get("bboxes", []),
                        source.get("bbox_labels", []),
                        overlay_bboxes=source_overlay.get("bboxes", []) if source_overlay else None,
                        overlay_bbox_labels=source_overlay.get("bbox_labels", []) if source_overlay else None,
                        **augmentation_options,
                    )
                else:
                    results = apply_augmentations(
                        source["image"], selected, **augmentation_options,
                    )
                for result_item in results:
                    if len(result_item) == 4:
                        technique, image, boxes, labels = result_item
                    else:
                        technique, image = result_item
                        boxes, labels = None, None
                    entry = save_generated(
                        image, source["name"], "traditional", [technique],
                        source["signature"], boxes, labels,
                    )
                    st.session_state.last_augmented.append(entry)
                    total_created += 1
            record_activity(f"Generated {total_created} traditional augmentation(s)")
            st.success(f"Created {total_created} transformed image(s).")
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
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in display:
                archive.write(item["path"], arcname=item["name"])
                annotation_path = Path(item["path"]).with_suffix(".txt")
                if annotation_path.is_file():
                    archive.write(annotation_path, arcname=annotation_path.name)
        st.download_button(
            "Download current batch (ZIP)", archive_buffer.getvalue(),
            "gateaugment_augmentation_batch.zip", "application/zip",
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
    variation_count = st.slider("Synthetic variations", 1, 4, 1)
    prompt = st.text_input("Prompt", "baggage x-ray scan, preserve the original object layout")
    fallback = st.checkbox("If model loading fails, use clearly labeled Albumentations fallback", value=True)
    if st.button("Generate with diffusion", type="primary"):
        with st.spinner("Loading model weights and generating an image. First use may download a large model..."):
            try:
                generated_images = generate_many_with_diffusion(
                    info["image"], model_id=model_id, prompt=prompt,
                    strength=strength, num_images=variation_count,
                )
                synthetic_entries = []
                generation_progress = st.progress(0, text="Saving synthetic variations…")
                for index, generated in enumerate(generated_images):
                    entry = save_generated(
                        generated, info["name"], "diffusion",
                        [f"Diffusers: {model_id}"], info["signature"],
                    )
                    synthetic_entries.append(entry)
                    generation_progress.progress(
                        (index + 1) / len(generated_images),
                        text=f"Saved variation {index + 1}/{len(generated_images)}",
                    )
                st.session_state.last_synthetic = synthetic_entries[-1]
                st.session_state.last_synthetic_batch = synthetic_entries
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
                        st.session_state.last_synthetic_batch = [entry]
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


def show_intelligence() -> None:
    st.markdown('<p class="eyebrow">Image intelligence · research models only</p>', unsafe_allow_html=True)
    st.title("Image intelligence")
    image = image_selector()
    if image is None:
        return
    rgb = image["image"]
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_intensity = float(np.mean(grayscale))
    contrast = float(np.std(grayscale))
    edges = cv2.Canny(grayscale, 50, 150)
    col1, col2 = st.columns(2)
    col1.image(rgb, caption=f"Input · {image['name']}", use_container_width=True)
    col2.image(edges, caption="Edge map · image processing only", use_container_width=True)
    first, second, third = st.columns(3)
    first.metric("Dimensions", f"{rgb.shape[1]} × {rgb.shape[0]}")
    second.metric("Mean grayscale intensity", f"{mean_intensity:.1f} / 255")
    third.metric("Grayscale contrast (σ)", f"{contrast:.1f}")
    histogram = np.histogram(grayscale, bins=32, range=(0, 256))[0]
    fig, axis = plt.subplots(figsize=(8, 2.5))
    axis.plot(np.linspace(0, 255, len(histogram)), histogram, color="#12a9bd")
    axis.set_title("Pixel intensity distribution")
    axis.set_xlabel("Grayscale intensity")
    axis.set_ylabel("Pixel count")
    st.pyplot(fig)
    plt.close(fig)
    result = st.session_state.training_result
    if result:
        model_variant = st.selectbox("Image-level classifier", ["baseline", "augmented"])
        if st.button("Run saved classifier"):
            try:
                import tensorflow as tf

                model_path = MODEL_DIR / f"{model_variant}_cnn.keras"
                if not model_path.is_file():
                    st.error(f"Saved classifier is unavailable: {model_path}")
                else:
                    model = tf.keras.models.load_model(model_path)
                    resized = cv2.resize(
                        rgb, (result["image_size"], result["image_size"]),
                        interpolation=cv2.INTER_AREA,
                    )
                    probabilities = model.predict(resized[None, ...], verbose=0)[0]
                    ranking = sorted(
                        zip(result["class_names"], probabilities.tolist()),
                        key=lambda item: item[1], reverse=True,
                    )
                    st.subheader("Experimental image-level classification")
                    st.dataframe(pd.DataFrame(ranking, columns=["Dataset class", "Model score"]), use_container_width=True)
                    st.warning("Classifier scores are research outputs, not validated threat alarms.")
                    log_event(DB_PATH, "classification", f"Ran {model_variant} classifier on {image['name']}", {
                        "scores": dict(ranking),
                    })
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                st.error(f"Classifier could not run: {exc}")
    else:
        st.info("Train and evaluate a CNN to enable saved-model image classification.")


def show_detection() -> None:
    st.markdown('<p class="eyebrow">Custom X-ray detector · no default weights</p>', unsafe_allow_html=True)
    st.title("Object detection")
    available, message = detection_availability()
    st.info(message)
    st.error(
        "No pretrained COCO weights are used here. Only load a trusted custom model trained "
        "for your X-ray dataset; detections are research outputs, not security decisions."
    )
    weight_dir = MODEL_DIR / "detection"
    weight_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(weight_dir.glob("*.pt"))
    if not available:
        st.warning("Object detection is unavailable until Ultralytics is installed.")
        return
    if not candidates:
        st.warning("No custom detector weights found. Place a trusted X-ray-trained `.pt` file in models/detection/.")
        return
    weights = st.selectbox("Custom detection weights", candidates, format_func=lambda path: path.name)
    image = image_selector()
    if image is None:
        return
    confidence = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
    if st.button("Run custom X-ray detector", type="primary"):
        with st.spinner("Loading the selected custom detector…"):
            try:
                result = detect_objects(image["image"], weights, confidence=confidence)
                detections = result["detections"]
                output_path = GENERATED_DIR / "detections"
                output_path.mkdir(exist_ok=True)
                output_file = output_path / f"detection_{uuid.uuid4().hex[:12]}.png"
                from PIL import Image

                Image.fromarray(result["annotated_image"]).save(output_file)
                result["annotated_path"] = str(output_file)
                result["source_name"] = image["name"]
                result["created_at"] = datetime.now(timezone.utc).isoformat()
                st.session_state.detection_result = result
                log_event(DB_PATH, "detection", f"Detected {len(detections)} object(s) in {image['name']}", {
                    "model": weights.name,
                    "classes": [item["class_name"] for item in detections],
                })
                record_activity(f"Custom detector processed {image['name']}")
                st.success(f"Model returned {len(detections)} detection(s).")
            except (RuntimeError, ValueError, OSError) as exc:
                st.error(f"Detection failed: {exc}")
    result = st.session_state.detection_result
    if result:
        st.image(result["annotated_image"], caption=f"Custom model output · {Path(result['model_path']).name}", use_container_width=True)
        detections = result["detections"]
        counts = pd.Series([item["class_name"] for item in detections], dtype="object").value_counts()
        st.metric("Detected object count", len(detections))
        st.dataframe(pd.DataFrame(detections), use_container_width=True)
        if len(counts):
            st.bar_chart(counts)
        output_path = Path(result["annotated_path"])
        st.download_button("Download annotated image", output_path.read_bytes(), output_path.name, "image/png")
        st.download_button(
            "Download detection report",
            json.dumps({key: value for key, value in result.items() if key != "annotated_image"}, indent=2).encode("utf-8"),
            "detection_report.json",
            "application/json",
        )


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
    scans = recent_scans(DB_PATH, limit=500)
    stats = event_summary(DB_PATH)
    detection_result = st.session_state.detection_result
    report = {
        "project_name": "GateAugment AI",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_images": [
            {key: value for key, value in item.items() if key not in {"image", "signature"}}
            for item in st.session_state.uploads
        ],
        "total_uploaded_images": sum(not item.get("demo", False) for item in st.session_state.uploads),
        "total_augmented_images": sum(
            item.get("kind") in {"traditional", "traditional_fallback"}
            for item in st.session_state.generated
        ),
        "total_synthetic_images": sum(
            item.get("kind") == "diffusion" for item in st.session_state.generated
        ),
        "total_scans": stats["scans"],
        "scan_history": scans,
        "dataset": str(st.session_state.get("selected_dataset", "Default dataset")),
        "experiment_id": training.get("experiment_id") if training else None,
        "detection_model": (
            Path(detection_result["model_path"]).name if detection_result else "Unavailable / not run"
        ),
        "generative_ai_models": sorted({
            technique.split(": ", 1)[1]
            for item in st.session_state.generated
            for technique in item.get("techniques", [])
            if technique.startswith("Diffusers: ")
        }),
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
    writer.writerow(["Scans completed", event_summary(DB_PATH)["scans"]])
    writer.writerow([
        "Augmented images",
        sum(item.get("kind") in {"traditional", "traditional_fallback"} for item in st.session_state.generated),
    ])
    writer.writerow(["All generated images", len(st.session_state.generated)])
    writer.writerow(["Synthetic diffusion images", sum(item.get("kind") == "diffusion" for item in st.session_state.generated)])
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
        writer.writerow(["Experiment ID", training.get("experiment_id")])
    else:
        writer.writerow(["No evaluated model metrics", "Train and evaluate both models to populate metrics."])
    detection = st.session_state.detection_result
    writer.writerow(["Detection model", Path(detection["model_path"]).name if detection else "Unavailable / not run"])
    diffusion_models = sorted({
        method.split(": ", 1)[1]
        for item in st.session_state.generated
        for method in item.get("techniques", [])
        if method.startswith("Diffusers: ")
    })
    writer.writerow(["Generative AI models", "; ".join(diffusion_models) if diffusion_models else "No diffusion model run"])
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
    report_data = json.loads(report_bytes.decode("utf-8"))
    try:
        pdf_bytes = build_pdf_report(report_data)
        (REPORT_DIR / "gateaugment_report.pdf").write_bytes(pdf_bytes)
    except RuntimeError as exc:
        pdf_bytes = None
        st.warning(str(exc))
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
    if pdf_bytes is not None:
        st.download_button(
            "Download PDF report", pdf_bytes,
            file_name="gateaugment_report.pdf", mime="application/pdf",
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
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(145deg, #e8f7ff 0%, #f6fbff 48%, #e8fff8 100%);
            color: #102a43;
        }
        .main h1, .main h2, .main h3, .main p, .main label {
            color: #102a43;
        }
        .main [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #c8e5f1;
            border-left: 5px solid #11a8c1;
            border-radius: 14px;
            box-shadow: 0 5px 18px rgba(17, 74, 103, 0.09);
        }
        .main [data-testid="stMetricValue"] {
            color: #087e8b;
        }
        .main [data-testid="stMetricLabel"] {
            color: #38556c;
        }
        .main [data-testid="stVegaLiteChart"],
        .main [data-testid="stArrowVegaLiteChart"],
        .main [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #d5e8f0;
            border-radius: 14px;
            padding: 8px;
            box-shadow: 0 5px 18px rgba(17, 74, 103, 0.06);
        }
        .main [data-testid="stAlert"] {
            border-radius: 12px;
        }
        .main .eyebrow {
            color: #008c9e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="eyebrow">Airport security intelligence control center</p>', unsafe_allow_html=True)
    st.title("GateAugment AI")
    st.caption("Local research workstation · not connected to airport screening hardware")
    training = st.session_state.training_result
    metrics = training["evaluation"] if training else None
    counts = event_summary(DB_PATH)
    synthetic_count = sum(
        item.get("kind") == "diffusion" for item in st.session_state.generated
    )
    trained_model_count = sum(
        (MODEL_DIR / f"{name}_cnn.keras").is_file()
        for name in ("baseline", "augmented")
    )
    cols = st.columns(7)
    uploaded_count = sum(not item.get("demo", False) for item in st.session_state.uploads)
    detection_count = counts["events"].get("detection", 0)
    cols[0].metric("Images uploaded", uploaded_count)
    cols[1].metric(
        "Augmented images",
        sum(item.get("kind") in {"traditional", "traditional_fallback"} for item in st.session_state.generated),
    )
    cols[2].metric("Scans completed", counts["scans"])
    cols[3].metric("Synthetic images", synthetic_count)
    cols[4].metric("Detection runs", detection_count)
    cols[5].metric("Trained models", f"{trained_model_count} / 2")
    cols[6].metric("Model accuracy", f"{metrics['augmented']['accuracy']:.3f}" if metrics else "Not evaluated")
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
            st.caption("Upload, scan or generate images to populate activity charts.")
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
    diffusion_ready, diffusion_message = diffusion_availability()
    detector_ready, detector_message = detection_availability()
    status_items = [
        ("Streamlit application", "Operational", True),
        ("Custom X-ray detector", "Available" if detector_ready else "Unavailable", detector_ready),
        ("Diffusion runtime", "Available" if diffusion_ready else "Unavailable", diffusion_ready),
        ("Research dataset", "Configured" if discover_datasets() else "Not configured", bool(discover_datasets())),
    ]
    st.subheader("System status")
    status_columns = st.columns(len(status_items))
    for column, (label, value, healthy) in zip(status_columns, status_items):
        column.metric(label, value, delta="Ready" if healthy else "Action required")
    st.caption(f"{detector_message} {diffusion_message}")
    st.warning("Research tool only. Validate every generated image and annotation; never use unreviewed synthetic imagery for operational security decisions.")
    scans = recent_scans(DB_PATH, limit=8)
    if scans:
        st.subheader("Recent scan history")
        st.dataframe(pd.DataFrame([
            {key: item[key] for key in (
                "created_at", "source", "filename", "operation",
                "status", "processing_ms",
            )}
            for item in scans
        ]), use_container_width=True)
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


def show_analytics() -> None:
    st.markdown('<p class="eyebrow">Smart analytics · observed project activity</p>', unsafe_allow_html=True)
    st.title("Smart analytics")
    summary = event_summary(DB_PATH)
    scans = recent_scans(DB_PATH, limit=500)
    events = recent_events(DB_PATH, limit=500)
    generated = st.session_state.generated
    upload_count = sum(not item.get("demo", False) for item in st.session_state.uploads)
    metric_cols = st.columns(6)
    metric_cols[0].metric("Image uploads", upload_count)
    metric_cols[1].metric(
        "Augmentations",
        sum(item.get("kind") in {"traditional", "traditional_fallback"} for item in generated),
    )
    metric_cols[2].metric("Diffusion outputs", sum(item.get("kind") == "diffusion" for item in generated))
    metric_cols[3].metric("Scan frames", summary["scans"])
    metric_cols[4].metric("Detection runs", summary["events"].get("detection", 0))
    metric_cols[5].metric("Training experiments", len(st.session_state.experiments))

    left, right = st.columns(2)
    with left:
        st.subheader("Scan throughput")
        if scans:
            frame = pd.DataFrame(scans)
            frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["created_at"]).set_index("created_at")
            st.line_chart(frame[["processing_ms"]].sort_index())
        else:
            st.info("No scanner frames have been processed.")
    with right:
        st.subheader("Generated content mix")
        mix = pd.Series(
            [item.get("kind", "unknown") for item in generated], dtype="object",
        ).value_counts()
        if len(mix):
            st.bar_chart(mix)
        else:
            st.info("No augmented or synthetic images have been generated.")
    result = st.session_state.training_result
    if result:
        st.subheader("Actual model performance")
        comparison = pd.DataFrame({
            "Baseline": [result["evaluation"]["baseline"][metric] for metric in ("accuracy", "precision", "recall", "f1")],
            "Augmented": [result["evaluation"]["augmented"][metric] for metric in ("accuracy", "precision", "recall", "f1")],
        }, index=["Accuracy", "Macro precision", "Macro recall", "Macro F1"])
        st.bar_chart(comparison)
    st.subheader("Experiment and processing activity")
    if events:
        st.dataframe(pd.DataFrame([
            {"Time (UTC)": item["created_at"], "Type": item["event_type"], "Description": item["description"]}
            for item in events
        ]), use_container_width=True)
    else:
        st.caption("Activity events appear after uploads, scans, augmentation, detection, or training.")
    st.download_button(
        "Export analytics (CSV)",
        pd.DataFrame([
            {"event_time": item["created_at"], "event_type": item["event_type"], "description": item["description"]}
            for item in events
        ]).to_csv(index=False).encode("utf-8-sig"),
        "gateaugment_analytics.csv",
        "text/csv",
    )


def show_settings() -> None:
    st.markdown('<p class="eyebrow">Workstation configuration</p>', unsafe_allow_html=True)
    st.title("Settings")
    st.subheader("Runtime components")
    modules = {
        "Streamlit": "streamlit",
        "OpenCV": "cv2",
        "Albumentations": "albumentations",
        "TensorFlow": "tensorflow",
        "Plotly (optional)": "plotly",
        "ReportLab (PDF)": "reportlab",
        "Ultralytics (custom YOLO)": "ultralytics",
        "PyTorch": "torch",
        "Diffusers": "diffusers",
    }
    from importlib.util import find_spec

    status_rows = []
    for label, module_name in modules.items():
        status_rows.append({"Component": label, "Status": "Installed" if find_spec(module_name) else "Unavailable"})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    st.subheader("Model and dataset locations")
    st.code(
        f"Dataset root: {DATASET_ROOT}\n"
        f"Training data: {TRAIN_DIR}\n"
        f"Test data: {TEST_DIR}\n"
        f"CNN models: {MODEL_DIR}\n"
        f"Custom detection weights: {MODEL_DIR / 'detection'}\n"
        f"Uploads: {ROOT / 'assets' / 'uploads'}\n"
        f"Scan database: {DB_PATH}",
        language="text",
    )
    st.markdown(
        "**Supported research data:** SIXray, OPIXray and PIDray can be organized under "
        "`dataset/<name>/train/<class>/` and `dataset/<name>/test/<class>/`. "
        "Download datasets from their official sources and follow their license terms."
    )
    st.warning(
        "No airport X-ray hardware interface is configured. Webcam input is a demonstration capture. "
        "Custom detector weights and diffusion outputs require independent validation."
    )
    if st.button("Reload persistent project data"):
        st.session_state.storage_initialized = False
        st.rerun()


PAGES = {
    "Security Intelligence Dashboard": show_dashboard,
    "Live Scanner": show_scanner,
    "Image upload": show_uploads,
    "Image Intelligence": show_intelligence,
    "Advanced Augmentation": show_augmentation,
    "Generative AI studio": show_generative_studio,
    "Object Detection": show_detection,
    "Model training": show_training,
    "Model evaluation": show_evaluation,
    "Results comparison": show_comparison,
    "Smart Analytics": show_analytics,
    "Reports and History": show_reports,
    "Settings": show_settings,
}

initialize_state()
with st.sidebar:
    st.markdown("## ✈ GateAugment AI")
    st.caption("AIRPORT SECURITY · IMAGING LAB")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Local-first research application")

PAGES[page]()
