"""
Real Data Fine-tuner - Phase 2
Improvements: AdamW + Cosine Warmup LR + Focal+Dice Loss + SE Attention + F1-based early stopping
"""

import os
import sys
import logging
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.litenet import LiteNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# LOSS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05,
                 focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal   = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.dice    = DiceLoss()
        self.focal_w = focal_weight
        self.dice_w  = dice_weight
        logger.info(f"CombinedLoss: focal={focal_weight} (a={alpha}, g={gamma}, ls={label_smoothing}) + dice={dice_weight}")

    def forward(self, logits, targets):
        return self.focal_w * self.focal(logits, targets) + self.dice_w * self.dice(logits, targets)


# ============================================================
# AUGMENTATION
# ============================================================

class PixelAugmentation:
    def __init__(self, spectral_noise_std=0.01, band_dropout_prob=0.05,
                 spatial_flip=True, apply_prob=0.4):
        self.spectral_noise_std = spectral_noise_std
        self.band_dropout_prob  = band_dropout_prob
        self.spatial_flip       = spatial_flip
        self.apply_prob         = apply_prob

    def __call__(self, sample):
        if torch.rand(1).item() > self.apply_prob:
            return sample
        spec = sample["spectrum"].clone()
        spat = sample["spatial"].clone()
        if self.spectral_noise_std > 0:
            spec = spec + torch.randn_like(spec) * self.spectral_noise_std
        if self.band_dropout_prob > 0:
            mask = torch.rand(spec.shape) > self.band_dropout_prob
            spec = spec * mask.float()
        if self.spatial_flip and torch.rand(1).item() > 0.5:
            spat = torch.flip(spat, dims=[2])
        if self.spatial_flip and torch.rand(1).item() > 0.5:
            spat = torch.flip(spat, dims=[1])
        return {"spectrum": spec, "spatial": spat, "label": sample["label"]}


class NoAugmentation:
    def __call__(self, sample):
        return sample


# ============================================================
# DATASET
# ============================================================

class PreprocessedDataset(Dataset):
    def __init__(self, pt_file_path, augmentation=None):
        pt_file_path = str(pt_file_path)
        logger.info(f"Loading: {pt_file_path}")
        if not Path(pt_file_path).exists():
            raise FileNotFoundError(f"File not found: {pt_file_path}")
        data = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        self.spectra   = data["spectra"].float()
        self.spatial   = data["spatial"].float()
        self.labels    = data["labels"].float()
        self.n_samples = len(self.labels)
        self.augmentation = augmentation if augmentation is not None else NoAugmentation()
        self.n_pos = int((self.labels == 1).sum().item())
        self.n_neg = int((self.labels == 0).sum().item())
        ratio = (self.n_pos / max(self.n_samples, 1)) * 100.0
        logger.info(f"  Loaded {self.n_samples:,} samples")
        logger.info(f"  Spectra: {tuple(self.spectra.shape)}")
        logger.info(f"  Spatial: {tuple(self.spatial.shape)}")
        logger.info(f"  Labels:  {tuple(self.labels.shape)} [{self.labels.min():.1f}, {self.labels.max():.1f}]")
        logger.info(f"  Label stats: neg={self.n_neg:,}, pos={self.n_pos:,}, pos_ratio={ratio:.2f}%")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {
            "spectrum": self.spectra[idx],
            "spatial":  self.spatial[idx],
            "label":    self.labels[idx],
        }
        return self.augmentation(sample)


# ============================================================
# METRICS
# ============================================================

class BinaryMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, probs, labels):
        preds   = (probs >= self.threshold).float()
        labels  = labels.float()
        self.tp += int(((preds == 1) & (labels == 1)).sum().item())
        self.fp += int(((preds == 1) & (labels == 0)).sum().item())
        self.fn += int(((preds == 0) & (labels == 1)).sum().item())
        self.tn += int(((preds == 0) & (labels == 0)).sum().item())

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        iou       = tp / max(tp + fp + fn, 1)
        return {"f1": round(f1, 4), "precision": round(precision, 4),
                "recall": round(recall, 4), "iou": round(iou, 4), "loss": 0.0}


# ============================================================
# HELPERS
# ============================================================

def make_weighted_sampler(labels):
    labels = labels.cpu()
    n_pos  = (labels == 1).sum().item()
    n_neg  = (labels == 0).sum().item()
    w_pos  = 1.0 / max(n_pos, 1)
    w_neg  = 1.0 / max(n_neg, 1)
    weights = torch.where(labels == 1,
                          torch.tensor(w_pos),
                          torch.tensor(w_neg)).double()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ============================================================
# FINETUNER
# ============================================================

class RealDataFinetuner:

    def __init__(self, config_path, pretrained_model_path):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ── Config sub-dicts (all saved as self.* for method access) ──
        train_cfg          = self.config.get("training", {})
        self.p2_cfg        = train_cfg.get("phase2", {})
        self.es_cfg        = train_cfg.get("early_stopping", {})
        self.loss_cfg      = self.p2_cfg.get("loss", {})
        self.sched_cfg     = self.p2_cfg.get("scheduler", {})
        self.aug_cfg       = self.p2_cfg.get("augmentation", {})

        # ── Training params ───────────────────────────────────────────
        self.epochs        = int(self.p2_cfg.get("epochs",        50))
        self.batch_size    = int(self.p2_cfg.get("batch_size",    128))
        self.lr            = float(self.p2_cfg.get("learning_rate", 5e-5))
        self.weight_decay  = float(self.p2_cfg.get("weight_decay",  1e-4))
        self.warmup_epochs = int(self.sched_cfg.get("warmup_epochs", 5))
        self.min_lr        = float(self.sched_cfg.get("min_lr",      1e-6))

        # ── Early stopping ────────────────────────────────────────────
        self.es_patience   = int(self.es_cfg.get("patience",    15))
        self.es_min_delta  = float(self.es_cfg.get("min_delta", 0.001))
        self.es_monitor    = str(self.es_cfg.get("monitor",     "val_f1"))

        # ── Augmentation ──────────────────────────────────────────────
        self.aug_enabled   = bool(self.aug_cfg.get("enabled",    True))
        self.aug_prob      = float(self.aug_cfg.get("apply_prob", 0.4))

        # ── Loss params ───────────────────────────────────────────────
        self.focal_alpha     = float(self.loss_cfg.get("focal_alpha",      0.75))
        self.focal_gamma     = float(self.loss_cfg.get("focal_gamma",      2.0))
        self.label_smoothing = float(self.loss_cfg.get("label_smoothing",  0.05))
        self.focal_weight    = float(self.loss_cfg.get("focal_weight",     0.7))
        self.dice_weight     = float(self.loss_cfg.get("dice_weight",      0.3))

        # ── Runtime state ─────────────────────────────────────────────
        self.pretrained_path = pretrained_model_path
        self.output_dir      = self.config.get("paths", {}).get(
                                   "phase2_output", "experiments/phase2_real")
        self.checkpoint_dir  = Path(self.output_dir)
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp         = self.device.type == "cuda"
        self.best_val_f1     = 0.0
        self.es_counter      = 0

        # ── Metrics ───────────────────────────────────────────────────
        self.train_metrics = BinaryMetrics(threshold=0.5)
        self.val_metrics   = BinaryMetrics(threshold=0.5)

        logger.info("=" * 70)
        logger.info("PHASE 2: REAL DATA FINE-TUNING (IMPROVED)")
        logger.info("=" * 70)
        logger.info(f"Device:        {self.device}")
        logger.info(f"Pretrained:    {self.pretrained_path}")
        logger.info(f"Epochs:        {self.epochs}")
        logger.info(f"Learning rate: {self.lr}")
        logger.info(f"Weight decay:  {self.weight_decay}")
        logger.info(f"Batch size:    {self.batch_size}")
        logger.info(f"Warmup epochs: {self.warmup_epochs}")

        # ── Build model ───────────────────────────────────────────────
        model_cfg = self.config.get("model", {})
        self.model = LiteNet(
            n_bands=model_cfg.get("input_bands", 204),
            spatial_channels=model_cfg.get("spatial_branch", {}).get("input_channels", 3),
            spectral_output_dim=model_cfg.get("spectral_branch", {}).get("channels", [None, None, 128])[-1],
            spatial_output_dim=model_cfg.get("spatial_branch", {}).get("channels", [None, 64])[-1],
            fusion_hidden_dim=model_cfg.get("fusion", {}).get("hidden_dim", 64),
            dropout=model_cfg.get("fusion", {}).get("dropout", 0.3),
            spectral_architecture=model_cfg.get("spectral_branch", {}).get("architecture", "simple"),
            spatial_input_size=3,
        )

        self._load_pretrained(pretrained_model_path)
        self.model = self.model.to(self.device)

        # ── Mixed precision ───────────────────────────────────────────
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        logger.info(f"Mixed precision: {'ENABLED' if self.use_amp else 'DISABLED'}")

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("[OK] Fine-tuner initialized")

    # ──────────────────────────────────────────────────────────
    def _load_pretrained(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ck["model_state_dict"], strict=False)
        logger.info("Pretrained weights loaded (strict=False)")
        if missing:
            logger.info(f"  New layers (random init): {len(missing)} keys")
            for k in missing: logger.info(f"    + {k}")
        if unexpected:
            logger.info(f"  Skipped old keys: {len(unexpected)} keys")
            for k in unexpected: logger.info(f"    - {k}")
        matched = len(ck["model_state_dict"]) - len(unexpected)
        logger.info(f"  Transferred: {matched}/{len(ck['model_state_dict'])} tensors from Phase 1")
        if matched == 0:
            raise RuntimeError("No weights transferred — architecture mismatch too severe.")

    def _maybe_freeze(self):
        if bool(self.p2_cfg.get("freeze_spectral_branch", False)):
            logger.info("Freezing spectral branch")
            for p in self.model.spectral_branch.parameters():
                p.requires_grad = False

    def _build_schedulers(self, total_epochs):
        # optimizer MUST exist before calling this
        warmup = self.warmup_epochs

        def warmup_lambda(epoch):
            return float(epoch + 1) / float(warmup) if epoch < warmup else 1.0

        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup, 1),
            eta_min=self.min_lr,
        )
        logger.info(f"Scheduler: LinearWarmup({warmup} ep) -> CosineAnnealing(T_max={total_epochs - warmup}, min_lr={self.min_lr})")

    def _step_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def _make_augmentation(self):
        if not self.aug_enabled:
            logger.info("Augmentation: DISABLED")
            return NoAugmentation()
        logger.info("Augmentation: ENABLED (Phase 2)")
        logger.info(f"  Apply prob: {self.aug_prob}")
        return PixelAugmentation(
            spectral_noise_std=self.aug_cfg.get("spectral_noise_std", 0.01),
            band_dropout_prob =self.aug_cfg.get("band_dropout_prob",  0.05),
            spatial_flip      =self.aug_cfg.get("spatial_flip",       True),
            apply_prob        =self.aug_prob,
        )

    def _save_best(self, epoch, val_metrics):
        path = self.checkpoint_dir / "best_model.pth"
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch":          epoch,
            "score":          val_metrics["f1"],
            "val_loss":       val_metrics["loss"],
            "val_f1":         val_metrics["f1"],
            "val_precision":  val_metrics["precision"],
            "val_recall":     val_metrics["recall"],
        }, path)
        logger.info(f"Saved best model: f1={val_metrics['f1']:.4f} -> {path}")

    def _check_early_stopping(self, val_f1):
        if val_f1 > self.best_val_f1 + self.es_min_delta:
            self.best_val_f1 = val_f1
            self.es_counter  = 0
            return False
        self.es_counter += 1
        logger.info(f"EarlyStopping: {self.es_counter}/{self.es_patience} epochs without improvement")
        return self.es_counter >= self.es_patience

    # ──────────────────────────────────────────────────────────
    def train_epoch(self, loader):
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Training", ncols=100)
        for batch in pbar:
            spectrum = batch["spectrum"].to(self.device, non_blocking=True)
            spatial  = batch["spatial"].to(self.device, non_blocking=True)
            labels   = batch["label"].to(self.device, non_blocking=True).unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(spectrum, spatial)
                    loss   = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(spectrum, spatial)
                loss   = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            probs = torch.sigmoid(logits.detach())
            self.train_metrics.update(probs, labels)
            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = self.train_metrics.compute()
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Validation", ncols=100):
            spectrum = batch["spectrum"].to(self.device, non_blocking=True)
            spatial  = batch["spatial"].to(self.device, non_blocking=True)
            labels   = batch["label"].to(self.device, non_blocking=True).unsqueeze(1)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(spectrum, spatial)
                    loss   = self.criterion(logits, labels)
            else:
                logits = self.model(spectrum, spatial)
                loss   = self.criterion(logits, labels)

            probs = torch.sigmoid(logits)
            self.val_metrics.update(probs, labels)
            total_loss += float(loss.item())

        metrics = self.val_metrics.compute()
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    # ──────────────────────────────────────────────────────────
    def finetune(self, train_data, val_data, use_preprocessed=True):
        if not use_preprocessed:
            raise ValueError("Only preprocessed .pt files supported.")

        # Datasets
        train_dataset = PreprocessedDataset(train_data, augmentation=self._make_augmentation())
        val_dataset   = PreprocessedDataset(val_data,   augmentation=NoAugmentation())

        if train_dataset.n_pos == 0:
            raise RuntimeError("No positive samples in training set!")

        # Loss
        self.criterion = CombinedLoss(
            alpha           = self.focal_alpha,
            gamma           = self.focal_gamma,
            label_smoothing = self.label_smoothing,
            focal_weight    = self.focal_weight,
            dice_weight     = self.dice_weight,
        )

        # Freeze layers if configured
        self._maybe_freeze()

        # ── Optimizer — must be created BEFORE _build_schedulers ──────
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Schedulers
        self._build_schedulers(self.epochs)

        # DataLoaders (num_workers=0 required on Windows)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        logger.info(f"Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
        logger.info(f"Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")
        logger.info(f"Fine-tuning for {self.epochs} epochs")
        logger.info("=" * 70)

        for epoch in range(self.epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"\nEPOCH {epoch+1}/{self.epochs}  [LR: {current_lr:.2e}]")
            logger.info("-" * 70)

            train_metrics = self.train_epoch(train_loader)
            logger.info(
                f"[TRAIN] Loss: {train_metrics['loss']:.4f} | "
                f"F1: {train_metrics['f1']:.4f} | "
                f"P: {train_metrics['precision']:.4f} | "
                f"R: {train_metrics['recall']:.4f}"
            )

            val_metrics = self.validate(val_loader)
            logger.info(
                f"[VAL]   Loss: {val_metrics['loss']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"P: {val_metrics['precision']:.4f} | "
                f"R: {val_metrics['recall']:.4f}"
            )

            self._step_scheduler(epoch)

            if val_metrics["f1"] > self.best_val_f1 + self.es_min_delta:
                self._save_best(epoch + 1, val_metrics)
                logger.info(f"*** NEW BEST F1: {val_metrics['f1']:.4f}")

            if self._check_early_stopping(val_metrics["f1"]):
                logger.info("Early stopping triggered.")
                break

        logger.info("=" * 70)
        logger.info(f"FINE-TUNING COMPLETE! Best val F1: {self.best_val_f1:.4f}")
        logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    finetuner = RealDataFinetuner(
        config_path="config/config.yaml",
        pretrained_model_path="experiments/phase1_synthetic/best_model.pth",
    )
    finetuner.finetune(
        train_data="data/processed/harmonized_204bands/agrifood_train_preprocessed.pt",
        val_data  ="data/processed/harmonized_204bands/agrifood_val_preprocessed.pt",
        use_preprocessed=True,
    )
