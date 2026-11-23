from __future__ import annotations

import cv2
import numpy as np
import albumentations as A

TECHNIQUES = {
    "Rotation": "Randomly rotates the image within the configured range.",
    "Horizontal flip": "Mirrors the image horizontally.",
    "Vertical flip": "Mirrors the image vertically; review whether this is realistic for the dataset.",
    "Brightness / contrast": "Adjusts brightness and contrast within configured bounds.",
    "Random scaling": "Scales the image while keeping the original canvas size.",
    "Translation": "Shifts image content on a fixed-size canvas.",
    "Crop": "Takes a random crop and resizes it back to the source dimensions.",
    "Occlusion": "Adds a small rectangular block to simulate partial visibility.",
    "Overlap composition": "Blends a second uploaded image only when dimensions match.",
}


def create_demo_image(width: int = 480, height: int = 320) -> np.ndarray:
    """Create an illustrative diagram-like image; it is not a dataset X-ray."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (16, 31, 42)
    cv2.rectangle(image, (45, 38), (width - 45, height - 38), (56, 76, 80), 3)
    cv2.rectangle(image, (70, 63), (width - 70, height - 63), (26, 43, 48), -1)
    cv2.ellipse(image, (width // 2, height // 2), (105, 66), -18, 0, 360, (125, 139, 128), -1)
    cv2.rectangle(image, (105, 128), (185, 171), (186, 172, 127), -1)
    cv2.circle(image, (260, 145), 27, (107, 151, 160), -1)
    cv2.line(image, (240, 171), (315, 188), (198, 185, 140), 9)
    cv2.putText(image, "ILLUSTRATIVE DEMO", (95, height - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (53, 213, 232), 2)
    return image


def _apply_one(image: np.ndarray, technique: str, *, angle: int, brightness: float,
               contrast: float, scale: float, translate_percent: int,
               overlay: np.ndarray | None) -> np.ndarray:
    height, width = image.shape[:2]
    if technique == "Rotation":
        limit = max(0, min(int(angle), 180))
        transform = A.Rotate(limit=(-limit, limit), p=1)
        return transform(image=image)["image"]
    if technique == "Horizontal flip":
        return A.HorizontalFlip(p=1)(image=image)["image"]
    if technique == "Vertical flip":
        return A.VerticalFlip(p=1)(image=image)["image"]
    if technique == "Brightness / contrast":
        b = max(-1.0, min(float(brightness), 1.0))
        c = max(-1.0, min(float(contrast), 1.0))
        return A.RandomBrightnessContrast(brightness_limit=(b, b), contrast_limit=(c, c), p=1)(image=image)["image"]
    if technique == "Random scaling":
        factor = max(0.5, min(float(scale), 2.0))
        scale_range = (min(factor, 1.0 / factor), max(factor, 1.0 / factor))
        transform = A.Affine(scale=scale_range, p=1)
        return transform(image=image)["image"]
    if technique == "Translation":
        pct = max(0, min(int(translate_percent), 40)) / 100
        transform = A.Affine(translate_percent={"x": (-pct, pct), "y": (-pct, pct)}, p=1)
        return transform(image=image)["image"]
    if technique == "Crop":
        crop_width = max(1, int(width * 0.8))
        crop_height = max(1, int(height * 0.8))
        x = np.random.randint(0, width - crop_width + 1)
        y = np.random.randint(0, height - crop_height + 1)
        cropped = image[y:y + crop_height, x:x + crop_width]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    if technique == "Occlusion":
        result = image.copy()
        block_width = max(1, int(width * 0.12))
        block_height = max(1, int(height * 0.12))
        x = np.random.randint(0, max(1, width - block_width + 1))
        y = np.random.randint(0, max(1, height - block_height + 1))
        patch = result[max(0, y - 3):min(height, y + block_height + 3),
                       max(0, x - 3):min(width, x + block_width + 3)]
        fill = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
        cv2.rectangle(result, (x, y), (x + block_width, y + block_height), fill, -1)
        return result
    if technique == "Overlap composition":
        if overlay is None:
            raise ValueError("Overlap composition requires a second uploaded image.")
        if overlay.shape != image.shape:
            raise ValueError("Overlap composition requires images with identical dimensions and channels.")
        return cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    raise ValueError(f"Unknown augmentation technique: {technique}")


def apply_augmentations(image: np.ndarray, techniques: list[str], count: int = 1, *,
                        angle: int = 15, brightness: float = 0.15,
                        contrast: float = 0.15, scale: float = 1.1,
                        translate_percent: int = 8,
                        overlay: np.ndarray | None = None) -> list[tuple[str, np.ndarray]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Augmentation expects an RGB image with three channels.")
    if not techniques:
        raise ValueError("Select at least one augmentation technique.")
    if count < 1:
        raise ValueError("Variation count must be at least one.")
    unknown = set(techniques) - set(TECHNIQUES)
    if unknown:
        raise ValueError(f"Unknown augmentation technique(s): {', '.join(sorted(unknown))}")
    output = []
    for technique in techniques:
        for _ in range(count):
            output.append((technique, _apply_one(
                image, technique, angle=angle, brightness=brightness,
                contrast=contrast, scale=scale, translate_percent=translate_percent,
                overlay=overlay,
            )))
    return output


def apply_augmentations_with_boxes(
    image: np.ndarray,
    techniques: list[str],
    bboxes: list[tuple[float, float, float, float]],
    bbox_labels: list[int],
    count: int = 1,
    *,
    angle: int = 15,
    brightness: float = 0.15,
    contrast: float = 0.15,
    scale: float = 1.1,
    translate_percent: int = 8,
    overlay: np.ndarray | None = None,
    overlay_bboxes: list[tuple[float, float, float, float]] | None = None,
    overlay_bbox_labels: list[int] | None = None,
) -> list[tuple[str, np.ndarray, list[tuple[float, float, float, float]], list[int]]]:
    """Apply transforms to YOLO-normalized boxes and their class labels alongside images."""
    if len(bboxes) != len(bbox_labels):
        raise ValueError("Each bounding box must have exactly one class label.")
    if "Overlap composition" in techniques:
        if overlay is None:
            raise ValueError("Overlap composition requires a second uploaded image.")
        if overlay.shape != image.shape:
            raise ValueError("Overlap composition requires images with identical dimensions and channels.")
        if overlay_bboxes is None or overlay_bbox_labels is None:
            raise ValueError("Upload YOLO annotations for both images before composing annotated images.")
        if len(overlay_bboxes) != len(overlay_bbox_labels):
            raise ValueError("Each overlay bounding box must have exactly one class label.")
    if not techniques or count < 1:
        raise ValueError("Select techniques and request at least one variation.")
    unknown = set(techniques) - set(TECHNIQUES)
    if unknown:
        raise ValueError(f"Unknown augmentation technique(s): {', '.join(sorted(unknown))}")

    height, width = image.shape[:2]
    crop_width, crop_height = max(1, int(width * 0.8)), max(1, int(height * 0.8))
    pct = max(0, min(int(translate_percent), 40)) / 100
    scale_factor = max(0.5, min(float(scale), 2.0))
    scale_range = (min(scale_factor, 1.0 / scale_factor), max(scale_factor, 1.0 / scale_factor))
    transforms = {
        "Rotation": A.Rotate(limit=(-max(0, min(int(angle), 180)), max(0, min(int(angle), 180))), p=1),
        "Horizontal flip": A.HorizontalFlip(p=1),
        "Vertical flip": A.VerticalFlip(p=1),
        "Brightness / contrast": A.RandomBrightnessContrast(
            brightness_limit=(max(-1.0, min(float(brightness), 1.0)),) * 2,
            contrast_limit=(max(-1.0, min(float(contrast), 1.0)),) * 2, p=1,
        ),
        "Random scaling": A.Affine(scale=scale_range, p=1),
        "Translation": A.Affine(
            translate_percent={"x": (-pct, pct), "y": (-pct, pct)}, p=1,
        ),
    }
    results = []
    for technique in techniques:
        for _ in range(count):
            if technique == "Overlap composition":
                merged_image = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
                merged_boxes = list(bboxes) + list(overlay_bboxes or [])
                merged_labels = list(bbox_labels) + list(overlay_bbox_labels or [])
                results.append((technique, merged_image, merged_boxes, merged_labels))
                continue
            if technique == "Occlusion":
                transformed_image = _apply_one(
                    image, technique, angle=angle, brightness=brightness,
                    contrast=contrast, scale=scale, translate_percent=translate_percent,
                    overlay=overlay,
                )
                results.append((technique, transformed_image, list(bboxes), list(bbox_labels)))
                continue
            if technique == "Brightness / contrast":
                transformed_image = transforms[technique](image=image)["image"]
                results.append((technique, transformed_image, list(bboxes), list(bbox_labels)))
                continue
            if technique == "Crop":
                x = np.random.randint(0, width - crop_width + 1)
                y = np.random.randint(0, height - crop_height + 1)
                transform = A.Compose(
                    [
                        A.Crop(x_min=x, y_min=y, x_max=x + crop_width, y_max=y + crop_height),
                        A.Resize(height=height, width=width),
                    ],
                    bbox_params=A.BboxParams(
                        format="yolo", label_fields=["bbox_labels"],
                        min_visibility=0.1, clip=True,
                    ),
                )
            else:
                transform = A.Compose(
                    [transforms[technique]],
                    bbox_params=A.BboxParams(
                        format="yolo", label_fields=["bbox_labels"],
                        min_visibility=0.1, clip=True,
                    ),
                )
            transformed = transform(
                image=image, bboxes=bboxes, bbox_labels=bbox_labels,
            )
            output_boxes = [tuple(float(value) for value in box) for box in transformed["bboxes"]]
            output_labels = [int(label) for label in transformed["bbox_labels"]]
            results.append((technique, transformed["image"], output_boxes, output_labels))
    return results
