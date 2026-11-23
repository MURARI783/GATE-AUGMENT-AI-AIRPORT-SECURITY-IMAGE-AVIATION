from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def ensure_project_dirs(root: Path) -> None:
    for relative in (
        "assets/uploads", "assets/annotations", "generated", "models", "reports",
        "dataset/train", "dataset/test",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def state_path(root: Path) -> Path:
    return root / "reports" / "project_state.json"


def load_project_state(root: Path) -> dict:
    path = state_path(root)
    if not path.exists():
        return {"uploads": [], "generated": [], "activities": [], "experiments": []}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read persistent project state at {path}: {exc}") from exc
    for key in ("uploads", "generated", "activities", "experiments"):
        if not isinstance(content.get(key), list):
            content[key] = []
    return content


def save_project_state(root: Path, state: dict) -> None:
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: state.get(key, [])
        for key in ("uploads", "generated", "activities", "experiments")
    }
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(serializable, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_uploaded_image(root: Path, filename: str, data: bytes) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    if not safe_name:
        safe_name = "uploaded_image.png"
    destination_dir = root / "assets" / "uploads"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{digest[:12]}_{safe_name}"
    if not destination.exists():
        temporary_path = destination.with_suffix(destination.suffix + ".tmp")
        temporary_path.write_bytes(data)
        temporary_path.replace(destination)
    return {
        "signature": digest,
        "name": Path(filename).name,
        "path": str(destination),
        "size_bytes": len(data),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def save_uploaded_annotation(root: Path, signature: str, text: str) -> str:
    destination_dir = root / "assets" / "annotations"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{signature[:16]}.txt"
    temporary_path = destination.with_suffix(".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(destination)
    return str(destination)


def load_rgb_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Saved image is missing or unreadable: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_saved_items(root: Path, entries: list[dict], *, is_upload: bool = False) -> list[dict]:
    loaded = []
    for entry in entries:
        path = Path(entry.get("path", ""))
        if not path.is_file():
            raise RuntimeError(f"Persistent project image is missing: {path}")
        image = load_rgb_image(str(path))
        item = dict(entry)
        item["image"] = image
        if is_upload:
            height, width = image.shape[:2]
            item.update({"width": width, "height": height, "channels": 3})
        loaded.append(item)
    return loaded
