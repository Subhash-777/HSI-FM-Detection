"""
Standalone pixel-level metric computation utilities.
"""
import numpy as np
import torch


def compute_all_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.float32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision   = tp / max(tp + fp, 1)
    recall      = tp / max(tp + fn, 1)
    f1          = 2 * precision * recall / max(precision + recall, 1e-9)
    iou         = tp / max(tp + fp + fn, 1)
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    fpr         = fp / max(fp + tn, 1)

    return {
        "f1": round(f1, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "iou": round(iou, 4),
        "accuracy": round(accuracy, 4), "specificity": round(specificity, 4),
        "fpr": round(fpr, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray,
                            metric: str = "f1", n_steps: int = 181) -> tuple:
    """Returns (best_threshold, best_score)."""
    thresholds = np.linspace(0.05, 0.95, n_steps)
    best_thresh, best_score = 0.5, 0.0
    for t in thresholds:
        m = compute_all_metrics(probs, labels, t)
        if m[metric] > best_score:
            best_score, best_thresh = m[metric], float(t)
    return best_thresh, best_score
