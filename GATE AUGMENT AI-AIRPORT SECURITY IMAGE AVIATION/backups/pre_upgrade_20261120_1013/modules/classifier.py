from __future__ import annotations

import json
from pathlib import Path

from modules.evaluation import evaluate_predictions


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is not installed. Install the project requirements to train CNN classifiers.") from exc
    return tf


def _build_model(tf, image_size: int, class_count: int, *, augmented: bool):
    layers = tf.keras.layers
    model_layers = [layers.Input(shape=(image_size, image_size, 3))]
    if augmented:
        model_layers.extend([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomContrast(0.15),
        ])
    model_layers.extend([
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(class_count, activation="softmax"),
    ])
    model = tf.keras.Sequential(model_layers)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(train_dir: str, test_dir: str, *, model_dir: str,
                       epochs: int = 5, batch_size: int = 16,
                       image_size: int = 128, validation_split: float = 0.2) -> dict:
    """Train baseline and on-the-fly-augmented models; evaluate both on one fixed test set."""
    tf = _tensorflow()
    train_path, test_path = Path(train_dir), Path(test_dir)
    if not train_path.is_dir() or not test_path.is_dir():
        raise ValueError("Training and test folders must both exist. See README for the required class-folder layout.")
    classes = sorted(p.name for p in train_path.iterdir() if p.is_dir())
    test_classes = sorted(p.name for p in test_path.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("At least two class subfolders are required under dataset/train.")
    if classes != test_classes:
        raise ValueError("Training and test folders must contain exactly the same class subfolders.")
    for root, names in [(train_path, classes), (test_path, test_classes)]:
        if any(
            not any(file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    for file in (root / name).rglob("*"))
            for name in names
        ):
            raise ValueError(f"Every class folder under {root} must contain at least one supported image.")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("Validation split must be between 0 and 1.")

    common = dict(
        directory=str(train_path), labels="inferred", label_mode="int",
        class_names=classes, image_size=(image_size, image_size),
        batch_size=batch_size, validation_split=validation_split, seed=17,
    )
    train_data = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_data = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    test_data = tf.keras.utils.image_dataset_from_directory(
        str(test_path), labels="inferred", label_mode="int", class_names=classes,
        image_size=(image_size, image_size), batch_size=batch_size, shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    train_data = train_data.prefetch(autotune)
    val_data = val_data.prefetch(autotune)
    test_data = test_data.prefetch(autotune)
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    evaluation = {}
    histories = {}
    for key, use_augmentation in [("baseline", False), ("augmented", True)]:
        model = _build_model(tf, image_size, len(classes), augmented=use_augmentation)
        history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=0)
        predictions = model.predict(test_data, verbose=0).argmax(axis=1)
        true_labels = []
        for _, labels in test_data:
            true_labels.extend(labels.numpy().tolist())
        evaluation[key] = evaluate_predictions(true_labels, predictions, classes)
        histories[key] = {
            metric: [float(value) for value in values]
            for metric, values in history.history.items()
        }
        model.save(model_root / f"{key}_cnn.keras")
        del model
    result = {
        "class_names": classes,
        "evaluation": evaluation,
        "history": histories,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "test_set": str(test_path),
        "augmentation_method": "Keras random horizontal flip, rotation, and contrast layers applied to training samples only",
    }
    (model_root / "latest_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
