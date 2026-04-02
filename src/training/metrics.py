"""
Training metrics for binary pixel classification.
Safe implementation (no broken dict/class indentation).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score


class PixelMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self._preds = []
        self._targs = []

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        # probs: (B,1) float [0,1], targets: (B,1) float {0,1}
        p = probs.detach().float().cpu().numpy().reshape(-1)
        t = targets.detach().float().cpu().numpy().reshape(-1)
        self._preds.append(p)
        self._targs.append(t)

    def compute(self):
        if len(self._preds) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "balanced_accuracy": 0.0}

        preds = np.concatenate(self._preds, axis=0)
        targs = np.concatenate(self._targs, axis=0).astype(int)

        pred_bin = (preds >= self.threshold).astype(int)

        precision = precision_score(targs, pred_bin, zero_division=0)
        recall = recall_score(targs, pred_bin, zero_division=0)
        f1 = f1_score(targs, pred_bin, zero_division=0)
        bal_acc = balanced_accuracy_score(targs, pred_bin)

        inter = np.sum((pred_bin == 1) & (targs == 1))
        union = np.sum((pred_bin == 1) | (targs == 1))
        iou = float(inter) / float(union + 1e-6)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "balanced_accuracy": float(bal_acc),
        }
