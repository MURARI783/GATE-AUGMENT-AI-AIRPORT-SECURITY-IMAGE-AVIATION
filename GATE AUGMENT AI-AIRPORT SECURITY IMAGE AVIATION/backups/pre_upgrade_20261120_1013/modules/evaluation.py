from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def evaluate_predictions(y_true, y_pred, class_names: list[str]) -> dict:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Evaluation requires non-empty true and predicted labels of equal length.")
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
