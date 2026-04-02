"""
Evaluate with Test-Time Augmentation (TTA)
Averages predictions over multiple augmented versions
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str, config: dict, device: torch.device):
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


def apply_tta(spectra, spatial, n_augs=4):
    """
    Generate TTA variants:
    1. Original
    2. Spatial flipped horizontally
    3. Spatial flipped vertically
    4. Spatial rotated 90°
    """
    variants = [(spectra, spatial)]  # Original
    
    # Horizontal flip
    variants.append((spectra, torch.flip(spatial, dims=[2])))
    
    # Vertical flip
    variants.append((spectra, torch.flip(spatial, dims=[1])))
    
    # Rotate 90°
    variants.append((spectra, torch.rot90(spatial, k=1, dims=(1, 2))))
    
    return variants[:n_augs]


def predict_with_tta(model, data_path: str, device: torch.device, batch_size: int = 128, n_augs: int = 4):
    """
    Run inference with TTA and average predictions
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    spectra = data["spectra"].float()
    spatial = data["spatial"].float()
    labels = data["labels"].float()
    
    n_samples = len(labels)
    logger.info(f"Running TTA inference on {n_samples:,} samples (x{n_augs} augmentations)...")
    
    all_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="TTA Inference", ncols=80):
            batch_spec = spectra[i:i+batch_size]
            batch_spat = spatial[i:i+batch_size]
            
            # Collect predictions for all augmentations
            aug_preds = []
            for spec, spat in zip(batch_spec, batch_spat):
                variants = apply_tta(spec.unsqueeze(0), spat.unsqueeze(0), n_augs)
                
                variant_preds = []
                for spec_var, spat_var in variants:
                    spec_var = spec_var.to(device)
                    spat_var = spat_var.to(device)
                    
                    logits = model(spec_var, spat_var)
                    probs = torch.sigmoid(logits).cpu()
                    variant_preds.append(probs)
                
                # Average predictions across augmentations
                avg_pred = torch.stack(variant_preds).mean(dim=0)
                aug_preds.append(avg_pred)
            
            all_probs.append(torch.cat(aug_preds, dim=0))
    
    all_probs = torch.cat(all_probs, dim=0).squeeze().numpy()
    labels = labels.numpy()
    
    return all_probs, labels


def compute_metrics(probs, labels, threshold=0.5):
    pred_bin = (probs >= threshold).astype(int)
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
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-augs", type=int, default=4, help="Number of TTA augmentations")
    parser.add_argument("--output", default="results/tta_evaluation.json")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    model = load_model(args.model, config, device)
    probs, labels = predict_with_tta(model, args.data, device, n_augs=args.n_augs)
    
    metrics = compute_metrics(probs, labels, args.threshold)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "model": str(args.model),
        "data": str(args.data),
        "n_samples": int(len(labels)),
        "n_augs": args.n_augs,
        "threshold": args.threshold,
        "metrics": metrics,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("TTA EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"Threshold: {args.threshold:.2f}")
    logger.info(f"F1:        {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"IoU:       {metrics['iou']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
