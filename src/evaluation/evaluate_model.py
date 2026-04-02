"""
Model evaluation with threshold tuning on validation set.
"""
import os, sys, logging, json
sys.path.insert(0, ".")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(self, model, device, output_dir="results/evaluation"):
        self.model      = model
        self.device     = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _collect_predictions(self, pt_path: str):
        """Load .pt file, run model, return (probs, labels) as numpy arrays."""
        import torch
        from pathlib import Path

        data    = torch.load(pt_path, map_location="cpu", weights_only=False)
        spectra = data["spectra"].float()
        spatial = data["spatial"].float()
        labels  = data["labels"].float().numpy()

        all_probs = []
        batch_size = 512
        self.model.eval()

        for i in range(0, len(spectra), batch_size):
            s = spectra[i:i+batch_size].to(self.device)
            p = spatial[i:i+batch_size].to(self.device)
            with torch.amp.autocast("cuda") if self.device.type == "cuda" else torch.no_grad():
                logits = self.model(s, p)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs), labels

    # ──────────────────────────────────────────────────────────────
    def tune_threshold(self, probs: np.ndarray, labels: np.ndarray):
        """
        Sweep thresholds from 0.05 to 0.95 and return the one
        that maximises F1 on the validation set.
        """
        thresholds  = np.linspace(0.05, 0.95, 181)
        best_thresh = 0.5
        best_f1     = 0.0
        curve       = []

        for t in thresholds:
            preds     = (probs >= t).astype(np.float32)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            curve.append({"threshold": float(t), "f1": float(f1),
                          "precision": float(prec), "recall": float(rec)})
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = float(t)

        logger.info(f"Threshold tuning complete → optimal={best_thresh:.3f}  F1={best_f1:.4f}")
        return best_thresh, curve

    # ──────────────────────────────────────────────────────────────
    def compute_metrics(self, probs, labels, threshold):
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            roc_curve, precision_recall_curve,
            confusion_matrix,
        )

        preds = (probs >= threshold).astype(np.float32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        iou       = tp / max(tp + fp + fn, 1)
        accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
        specificity = tn / max(tn + fp, 1)

        auc_roc = float(roc_auc_score(labels, probs))
        auc_pr  = float(average_precision_score(labels, probs))

        fpr, tpr, roc_thresh = roc_curve(labels, probs)
        pre, rec, pr_thresh  = precision_recall_curve(labels, probs)
        cm = confusion_matrix(labels, preds).tolist()

        return {
            "threshold":   threshold,
            "f1":          round(f1, 4),
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "iou":         round(iou, 4),
            "accuracy":    round(accuracy, 4),
            "specificity": round(specificity, 4),
            "auc_roc":     round(auc_roc, 4),
            "auc_pr":      round(auc_pr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "confusion_matrix": cm,
            "roc_curve":  {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve":   {"precision": pre.tolist(), "recall": rec.tolist()},
        }

    # ──────────────────────────────────────────────────────────────
    def run(self, val_pt: str, test_pt: str = None):
        logger.info("Step 1/3: Collecting validation predictions...")
        val_probs, val_labels = self._collect_predictions(val_pt)

        logger.info("Step 2/3: Threshold tuning on validation set...")
        opt_thresh, thresh_curve = self.tune_threshold(val_probs, val_labels)

        logger.info("Step 3/3: Computing final metrics...")
        val_metrics = self.compute_metrics(val_probs, val_labels, opt_thresh)

        results = {
            "optimal_threshold": opt_thresh,
            "val_metrics":       val_metrics,
            "threshold_curve":   thresh_curve,
        }

        if test_pt and test_pt != val_pt:
            logger.info("  Computing test metrics...")
            test_probs, test_labels = self._collect_predictions(test_pt)
            results["test_metrics"] = self.compute_metrics(test_probs, test_labels, opt_thresh)

        # Save JSON
        out = self.output_dir / "evaluation_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {out}")

        return results
