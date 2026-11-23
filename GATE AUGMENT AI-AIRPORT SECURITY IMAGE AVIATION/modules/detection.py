from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np


def detection_availability() -> tuple[bool, str]:
    if find_spec("ultralytics") is None:
        return False, "Ultralytics is not installed. Install the optional detection dependencies to enable custom YOLO models."
    return True, "Ultralytics is installed. A trained X-ray detection weights file is still required."


def detect_objects(image: np.ndarray, weights_path: Path, *, confidence: float = 0.25) -> dict:
    available, message = detection_availability()
    if not available:
        raise RuntimeError(message)
    if not weights_path.is_file() or weights_path.suffix.lower() not in {".pt", ".onnx"}:
        raise ValueError("Select an existing custom .pt or .onnx detector weights file.")
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        prediction = model.predict(image, conf=float(confidence), verbose=False)[0]
        annotated = prediction.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        boxes = []
        if prediction.boxes is not None:
            names = prediction.names
            for box in prediction.boxes:
                class_id = int(box.cls.item())
                boxes.append({
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id)),
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                })
        return {
            "annotated_image": annotated,
            "detections": boxes,
            "model_path": str(weights_path),
            "model_names": prediction.names,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Custom detector could not load or run: {exc}") from exc
