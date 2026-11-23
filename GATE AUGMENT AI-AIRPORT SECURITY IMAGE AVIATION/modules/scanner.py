from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def decode_video(
    data: bytes, suffix: str, *, max_frames: int = 120,
) -> tuple[list[np.ndarray], float]:
    if suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format. Use: {', '.join(sorted(VIDEO_EXTENSIONS))}.")
    if not data or len(data) > 100 * 1024 * 1024:
        raise ValueError("Video must be non-empty and no larger than 100 MB.")
    temporary_path = Path.cwd() / "generated" / f"scanner_input_{time.time_ns()}{suffix.lower()}"
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_path.write_bytes(data)
        capture = cv2.VideoCapture(str(temporary_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video file.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = []
        index = 0
        sampling_interval = max(1, round(fps)) if fps else 1
        while len(frames) < max(1, min(int(max_frames), 300)):
            ok, frame = capture.read()
            if not ok:
                break
            if index % sampling_interval == 0:
                height, width = frame.shape[:2]
                max_dimension = 1280
                if max(height, width) > max_dimension:
                    factor = max_dimension / max(height, width)
                    frame = cv2.resize(
                        frame, (int(width * factor), int(height * factor)),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        capture.release()
        if not frames:
            raise ValueError("No readable frames were found in this video.")
        return frames, fps if fps > 0 else 0.0
    finally:
        temporary_path.unlink(missing_ok=True)


def process_scan(image: np.ndarray, operation: str) -> tuple[np.ndarray, float]:
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Scanner requires a non-empty image.")
    started = time.perf_counter()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    if operation == "Grayscale":
        output = gray
    elif operation == "Contrast enhancement":
        output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    elif operation == "Edge detection":
        output = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    elif operation == "Sharpen":
        output = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    elif operation == "Enhance":
        output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    else:
        raise ValueError(f"Unknown scanner operation: {operation}")
    return output, (time.perf_counter() - started) * 1000


def validate_processed_image(image: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if image.size == 0 or image.shape[:2] != expected_shape[:2]:
        raise ValueError("Processed frame failed validation.")
    if image.dtype != np.uint8 or not np.isfinite(image).all():
        raise ValueError("Processed frame must contain finite uint8 pixel values.")
