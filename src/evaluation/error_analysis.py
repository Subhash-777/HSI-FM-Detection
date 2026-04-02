"""
Error analysis: hard positives, hard negatives, score distributions.
"""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorAnalyzer:

    def __init__(self, output_dir="results/evaluation"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def analyze(self, probs: np.ndarray, labels: np.ndarray, threshold: float):
        preds = (probs >= threshold).astype(np.float32)

        tp_mask = (preds == 1) & (labels == 1)
        fp_mask = (preds == 1) & (labels == 0)
        fn_mask = (preds == 0) & (labels == 1)
        tn_mask = (preds == 0) & (labels == 0)

        # Confidence stats per category
        def stats(mask):
            p = probs[mask]
            if len(p) == 0:
                return {"count": 0, "mean_conf": 0.0, "min_conf": 0.0, "max_conf": 0.0}
            return {"count": int(len(p)), "mean_conf": float(p.mean()),
                    "min_conf": float(p.min()), "max_conf": float(p.max())}

        # Hard examples: FP with high confidence, FN with high confidence
        hard_fp_thresh = 0.8
        hard_fn_thresh = 0.2
        hard_fp = int(((probs >= hard_fp_thresh) & (labels == 0)).sum())
        hard_fn = int(((probs <= hard_fn_thresh) & (labels == 1)).sum())

        report = {
            "threshold": threshold,
            "tp_stats": stats(tp_mask),
            "fp_stats": stats(fp_mask),
            "fn_stats": stats(fn_mask),
            "tn_stats": stats(tn_mask),
            "hard_false_positives": hard_fp,
            "hard_false_negatives": hard_fn,
            "pos_score_mean": float(probs[labels == 1].mean()),
            "neg_score_mean": float(probs[labels == 0].mean()),
            "score_separation": float(probs[labels == 1].mean() - probs[labels == 0].mean()),
        }

        import json
        with open(self.out / "error_analysis.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Score separation: {report['score_separation']:.4f}")
        logger.info(f"Hard FPs (conf>0.8): {hard_fp}  |  Hard FNs (conf<0.2): {hard_fn}")
        return report
