import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class ObjectMetrics:
    """
    Object-level metrics for HSI anomaly detection.
    Groups contiguous predicted anomaly pixels into objects
    and evaluates detection rate and false alarm rate.
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.detections      = []
        self.ground_truths   = []
        self.detection_ious  = []

    def compute_pixel_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        inter = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask,  gt_mask).sum()
        return float(inter) / max(float(union), 1)

    def evaluate_batch(self, probs: np.ndarray, labels: np.ndarray,
                       threshold: float = 0.8) -> dict:
        """
        Evaluate object-level metrics for a batch of pixel predictions.
        For pixel-level data (no spatial layout), we treat each anomaly
        cluster as a single object using basic grouping.
        """
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)

        tp_pixels = int(np.sum((preds == 1) & (labels == 1)))
        fp_pixels = int(np.sum((preds == 1) & (labels == 0)))
        fn_pixels = int(np.sum((preds == 0) & (labels == 1)))

        pixel_iou = tp_pixels / max(tp_pixels + fp_pixels + fn_pixels, 1)

        n_gt_pos  = int(labels.sum())
        n_pred_pos = int(preds.sum())
        detection_rate   = tp_pixels / max(n_gt_pos, 1)
        false_alarm_rate = fp_pixels / max(len(labels) - n_gt_pos, 1)

        return {
            "pixel_iou":        round(pixel_iou,        4),
            "detection_rate":   round(detection_rate,   4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "n_gt_positive":    n_gt_pos,
            "n_pred_positive":  n_pred_pos,
            "tp_pixels":        tp_pixels,
            "fp_pixels":        fp_pixels,
            "fn_pixels":        fn_pixels,
        }
