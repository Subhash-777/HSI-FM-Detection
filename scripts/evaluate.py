"""
Master evaluation script: threshold tuning → metrics → visualizations
Usage: python scripts/evaluate.py
"""
import os, sys
sys.path.insert(0, ".")

import torch
import numpy as np
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.litenet import LiteNet
from src.evaluation.evaluate_model import ModelEvaluator
from src.evaluation.visualize_results import ResultVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────
    model_cfg = config.get("model", {})
    model = LiteNet(
        n_bands=model_cfg.get("input_bands", 204),
        spatial_channels=model_cfg.get("spatial_branch", {}).get("input_channels", 3),
        spectral_output_dim=model_cfg.get("spectral_branch", {}).get("channels", [None, None, 128])[-1],
        spatial_output_dim=model_cfg.get("spatial_branch", {}).get("channels", [None, 64])[-1],
        fusion_hidden_dim=model_cfg.get("fusion", {}).get("hidden_dim", 64),
        dropout=0.0,  # disable dropout at eval
        spectral_architecture=model_cfg.get("spectral_branch", {}).get("architecture", "simple"),
        spatial_input_size=3,
    )

    ckpt_path = "experiments/phase2_real/best_model.pth"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()
    logger.info(f"Loaded model from {ckpt_path} (trained epoch {ck.get('epoch', '?')})")

    # ── Run evaluation with threshold tuning ───────────────────────
    evaluator = ModelEvaluator(model, device, output_dir="results/evaluation")
    results   = evaluator.run(
        val_pt  ="data/processed/harmonized_204bands/agrifood_val_preprocessed.pt",
        test_pt ="data/processed/harmonized_204bands/agrifood_val_preprocessed.pt",  # replace with test if available
    )

    # ── Visualize ──────────────────────────────────────────────────
    viz = ResultVisualizer(output_dir="results/plots")
    viz.plot_all(results)

    logger.info("=" * 60)
    logger.info(f"Optimal threshold : {results['optimal_threshold']:.4f}")
    logger.info(f"Val  F1           : {results['val_metrics']['f1']:.4f}")
    logger.info(f"Val  Precision    : {results['val_metrics']['precision']:.4f}")
    logger.info(f"Val  Recall       : {results['val_metrics']['recall']:.4f}")
    logger.info(f"Val  AUC-ROC      : {results['val_metrics']['auc_roc']:.4f}")
    logger.info(f"Val  AUC-PR       : {results['val_metrics']['auc_pr']:.4f}")
    logger.info("=" * 60)
