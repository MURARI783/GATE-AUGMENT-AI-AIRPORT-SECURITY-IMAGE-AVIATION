from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_upload(filename: str, data: bytes) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported image format. Upload JPG, JPEG, or PNG.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("Image exceeds the 25 MB upload limit.")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError("The file extension is supported, but its contents are not a readable image.")


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Could not decode image data.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_metadata(name: str, size_bytes: int, image: np.ndarray) -> dict:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel color image.")
    height, width = image.shape[:2]
    return {
        "name": name,
        "size_bytes": int(size_bytes),
        "width": int(width),
        "height": int(height),
        "channels": 3,
    }


def resize_for_model(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize RGB image to (width, height) using area or cubic interpolation."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Model image dimensions must be positive.")
    interpolation = cv2.INTER_AREA if image.shape[1] > width or image.shape[0] > height else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)
