"""
Evaluation visualizations: ROC, PR curve, confusion matrix,
threshold sweep, probability histogram.
"""
import os, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultVisualizer:

    def __init__(self, output_dir="results/plots"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    def plot_all(self, results: dict):
        vm = results["val_metrics"]
        tc = results["threshold_curve"]

        self.plot_roc_curve(vm)
        self.plot_pr_curve(vm)
        self.plot_confusion_matrix(vm)
        self.plot_threshold_sweep(tc, results["optimal_threshold"])
        self.plot_summary_dashboard(results)
        logger.info(f"All plots saved → {self.out}/")

    # ──────────────────────────────────────────────────────────────
    def plot_roc_curve(self, metrics: dict):
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        auc = metrics["auc_roc"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        plt.tight_layout()
        fig.savefig(self.out / "roc_curve.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: roc_curve.png  (AUC={auc:.4f})")

    # ──────────────────────────────────────────────────────────────
    def plot_pr_curve(self, metrics: dict):
        pre = metrics["pr_curve"]["precision"]
        rec = metrics["pr_curve"]["recall"]
        auc = metrics["auc_pr"]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, pre, color="#4CAF50", lw=2, label=f"AP = {auc:.4f}")
        ax.fill_between(rec, pre, alpha=0.1, color="#4CAF50")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        plt.tight_layout()
        fig.savefig(self.out / "pr_curve.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: pr_curve.png  (AP={auc:.4f})")

    # ──────────────────────────────────────────────────────────────
    def plot_confusion_matrix(self, metrics: dict):
        cm  = np.array(metrics["confusion_matrix"])
        tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
        labels_txt = [["TN", "FP"], ["FN", "TP"]]
        vals       = [[tn, fp], [fn, tp]]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred NEG", "Pred POS"], fontsize=11)
        ax.set_yticklabels(["True NEG", "True POS"], fontsize=11)

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = vals[i][j] / max(total, 1) * 100
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{labels_txt[i][j]}\n{vals[i][j]:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=10, color=color, fontweight="bold")

        ax.set_title(
            f"Confusion Matrix\nF1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}",
            fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        fig.savefig(self.out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: confusion_matrix.png")

    # ──────────────────────────────────────────────────────────────
    def plot_threshold_sweep(self, curve: list, optimal: float):
        thresholds = [c["threshold"] for c in curve]
        f1s        = [c["f1"]        for c in curve]
        precs      = [c["precision"] for c in curve]
        recs       = [c["recall"]    for c in curve]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, f1s,   color="#E91E63", lw=2.5, label="F1")
        ax.plot(thresholds, precs, color="#2196F3", lw=1.5, label="Precision", alpha=0.8)
        ax.plot(thresholds, recs,  color="#4CAF50", lw=1.5, label="Recall",    alpha=0.8)
        best_f1 = max(f1s)
        ax.axvline(x=optimal, color="#FF5722", lw=2, linestyle="--",
                   label=f"Optimal t={optimal:.3f}  F1={best_f1:.4f}")
        ax.set_xlabel("Classification Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Threshold vs. Metrics", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        plt.tight_layout()
        fig.savefig(self.out / "threshold_sweep.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: threshold_sweep.png  (optimal={optimal:.3f})")

    # ──────────────────────────────────────────────────────────────
    def plot_summary_dashboard(self, results: dict):
        vm  = results["val_metrics"]
        tc  = results["threshold_curve"]
        opt = results["optimal_threshold"]

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("HSI Anomaly Detection — Evaluation Dashboard",
                     fontsize=16, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr = vm["roc_curve"]["fpr"]; tpr = vm["roc_curve"]["tpr"]
        ax1.plot(fpr, tpr, "#2196F3", lw=2, label=f"AUC={vm['auc_roc']:.3f}")
        ax1.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
        ax1.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
        ax1.set_title("ROC Curve", fontweight="bold"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        # 2. PR Curve
        ax2 = fig.add_subplot(gs[0, 1])
        pre = vm["pr_curve"]["precision"]; rec = vm["pr_curve"]["recall"]
        ax2.plot(rec, pre, "#4CAF50", lw=2, label=f"AP={vm['auc_pr']:.3f}")
        ax2.fill_between(rec, pre, alpha=0.1, color="#4CAF50")
        ax2.set_title("PR Curve", fontweight="bold"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        # 3. Threshold sweep
        ax3 = fig.add_subplot(gs[0, 2])
        th  = [c["threshold"] for c in tc]
        f1s = [c["f1"]        for c in tc]
        ax3.plot(th, f1s, "#E91E63", lw=2)
        ax3.axvline(x=opt, color="#FF5722", lw=2, linestyle="--",
                    label=f"opt={opt:.3f}")
        ax3.set_title("Threshold Sweep", fontweight="bold")
        ax3.set_xlabel("Threshold"); ax3.set_ylabel("F1")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

        # 4. Confusion matrix
        ax4 = fig.add_subplot(gs[1, 0])
        cm  = np.array([[vm["tn"], vm["fp"]], [vm["fn"], vm["tp"]]])
        im  = ax4.imshow(cm, cmap="Blues", interpolation="nearest")
        ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
        ax4.set_xticklabels(["Pred NEG","Pred POS"]); ax4.set_yticklabels(["True NEG","True POS"])
        lbls = [["TN","FP"],["FN","TP"]]
        vals = [[vm["tn"],vm["fp"]],[vm["fn"],vm["tp"]]]
        for i in range(2):
            for j in range(2):
                c = "white" if cm[i,j] > cm.max()/2 else "black"
                ax4.text(j, i, f"{lbls[i][j]}\n{vals[i][j]:,}", ha="center", va="center",
                         fontsize=9, color=c, fontweight="bold")
        ax4.set_title("Confusion Matrix", fontweight="bold")

        # 5. Metrics bar chart
        ax5 = fig.add_subplot(gs[1, 1])
        metric_names  = ["F1", "Precision", "Recall", "IoU", "AUC-ROC", "AUC-PR"]
        metric_values = [vm["f1"], vm["precision"], vm["recall"],
                         vm["iou"], vm["auc_roc"], vm["auc_pr"]]
        colors = ["#E91E63","#2196F3","#4CAF50","#FF9800","#9C27B0","#00BCD4"]
        bars = ax5.bar(metric_names, metric_values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, metric_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax5.set_ylim(0, 1.15); ax5.set_title("Metric Summary", fontweight="bold")
        ax5.tick_params(axis="x", rotation=30); ax5.grid(True, alpha=0.2, axis="y")

        # 6. Key stats text box
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        txt = (
            f"EVALUATION RESULTS\n"
            f"{'─'*28}\n"
            f"Optimal Threshold : {opt:.4f}\n\n"
            f"F1 Score          : {vm['f1']:.4f}\n"
            f"Precision         : {vm['precision']:.4f}\n"
            f"Recall            : {vm['recall']:.4f}\n"
            f"IoU               : {vm['iou']:.4f}\n"
            f"Accuracy          : {vm['accuracy']:.4f}\n"
            f"Specificity       : {vm['specificity']:.4f}\n\n"
            f"AUC-ROC           : {vm['auc_roc']:.4f}\n"
            f"AUC-PR            : {vm['auc_pr']:.4f}\n\n"
            f"TP={vm['tp']:,}  FP={vm['fp']:,}\n"
            f"FN={vm['fn']:,}  TN={vm['tn']:,}"
        )
        ax6.text(0.05, 0.95, txt, transform=ax6.transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                           edgecolor="#BDBDBD", linewidth=1.5))

        fig.savefig(self.out / "evaluation_dashboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: evaluation_dashboard.png")
