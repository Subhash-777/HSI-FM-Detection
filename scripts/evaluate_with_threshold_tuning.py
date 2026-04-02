"""
Evaluate model with automatic threshold tuning
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, ".")

from src.models.litenet import LiteNet
from src.training.metrics import PixelMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str, config: dict, device: torch.device):
    """Load trained model"""
    cfg = config["model"]
    model = LiteNet(
        n_bands=cfg["input_bands"],
        spatial_channels=cfg["spatial_branch"]["input_channels"],
        spectral_output_dim=cfg["spectral_branch"]["channels"][-1],
        spatial_output_dim=cfg["spatial_branch"]["channels"][-1],
        fusion_hidden_dim=cfg["fusion"]["hidden_dim"],
        dropout=cfg["fusion"]["dropout"],
        spectral_architecture=cfg["spectral_branch"]["architecture"],
        spatial_input_size=3,
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded from: {model_path}")
    return model


def predict_on_dataset(model, data_path: str, device: torch.device, batch_size: int = 256):
    """
    Run inference and return (probabilities, labels)
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    spectra = data["spectra"].float()
    spatial = data["spatial"].float()
    labels = data["labels"].float()
    
    n_samples = len(labels)
    logger.info(f"Running inference on {n_samples:,} samples...")
    
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Inference", ncols=80):
            batch_spec = spectra[i:i+batch_size].to(device)
            batch_spat = spatial[i:i+batch_size].to(device)
            
            logits = model(batch_spec, batch_spat)
            probs = torch.sigmoid(logits).cpu().squeeze()
            all_probs.append(probs)
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    labels = labels.numpy()
    
    return all_probs, labels


def find_best_threshold(probs: np.ndarray, labels: np.ndarray, optimize_for: str = "f1",
                       start: float = 0.3, end: float = 0.7, step: float = 0.05):
    """
    Grid search for best threshold
    """
    logger.info(f"Searching best threshold (optimize for: {optimize_for})")
    logger.info(f"  Range: [{start}, {end}], step: {step}")
    
    thresholds = np.arange(start, end + step/2, step)
    results = []
    
    for thr in thresholds:
        pred_bin = (probs >= thr).astype(int)
        labels_int = labels.astype(int)
        
        tp = np.sum((pred_bin == 1) & (labels_int == 1))
        fp = np.sum((pred_bin == 1) & (labels_int == 0))
        fn = np.sum((pred_bin == 0) & (labels_int == 1))
        tn = np.sum((pred_bin == 0) & (labels_int == 0))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        inter = np.sum((pred_bin == 1) & (labels_int == 1))
        union = np.sum((pred_bin == 1) | (labels_int == 1))
        iou = inter / max(union, 1)
        
        results.append({
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
        })
    
    # Find best
    if optimize_for == "f1":
        best = max(results, key=lambda x: x["f1"])
    elif optimize_for == "precision":
        best = max(results, key=lambda x: x["precision"])
    elif optimize_for == "recall":
        best = max(results, key=lambda x: x["recall"])
    elif optimize_for == "iou":
        best = max(results, key=lambda x: x["iou"])
    else:
        best = max(results, key=lambda x: x["f1"])
    
    logger.info(f"✓ Best threshold: {best['threshold']:.2f}")
    logger.info(f"  F1: {best['f1']:.4f} | P: {best['precision']:.4f} | R: {best['recall']:.4f} | IoU: {best['iou']:.4f}")
    
    return best, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--data", required=True, help="Path to test data (.pt)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="results/threshold_tuning.json")
    parser.add_argument("--optimize-for", default="f1", choices=["f1", "precision", "recall", "iou"])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load model
    model = load_model(args.model, config, device)
    
    # Run inference
    probs, labels = predict_on_dataset(model, args.data, device)
    
    # Threshold tuning
    eval_cfg = config["evaluation"]["threshold_search"]
    best, all_results = find_best_threshold(
        probs, labels,
        optimize_for=args.optimize_for,
        start=eval_cfg["start"],
        end=eval_cfg["end"],
        step=eval_cfg["step"],
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "model": str(args.model),
        "data": str(args.data),
        "n_samples": int(len(labels)),
        "n_positives": int(labels.sum()),
        "best_threshold": best,
        "all_thresholds": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_path}")
    
    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("THRESHOLD TUNING RESULTS")
    logger.info("="*70)
    logger.info(f"{'Threshold':<12} {'F1':<10} {'Precision':<12} {'Recall':<10} {'IoU':<10}")
    logger.info("-"*70)
    for r in all_results:
        logger.info(
            f"{r['threshold']:<12.2f} {r['f1']:<10.4f} {r['precision']:<12.4f} "
            f"{r['recall']:<10.4f} {r['iou']:<10.4f}"
        )
    logger.info("="*70)


if __name__ == "__main__":
    main()
