"""
Phase 1: Train on synthetic data with optional augmentation
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml
from tqdm import tqdm

from ..models.litenet import LiteNet
from .losses import CombinedLoss
from .metrics import PixelMetrics
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from .augmentations import PixelAugmentation, NoAugmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PreprocessedDataset(Dataset):
    """Fast dataset from preprocessed .pt files with augmentation support"""

    def __init__(self, pt_file_path: str, augmentation=None):
        p = Path(pt_file_path)
        logger.info(f"Loading: {p}")
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        data = torch.load(p, map_location="cpu", weights_only=False)
        self.spectra = data["spectra"].float()
        self.spatial = data["spatial"].float()
        self.labels = data["labels"].float()
        self.n_samples = int(data["n_samples"])
        
        self.augmentation = augmentation if augmentation is not None else NoAugmentation()

        logger.info(f"✓ Loaded {self.n_samples:,} samples")
        logger.info(f"  Spectra: {tuple(self.spectra.shape)}")
        logger.info(f"  Spatial: {tuple(self.spatial.shape)}")
        logger.info(f"  Labels:  {tuple(self.labels.shape)} [{self.labels.min().item():.1f}, {self.labels.max().item():.1f}]")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {
            "spectrum": self.spectra[idx],
            "spatial": self.spatial[idx],
            "label": self.labels[idx],
        }
        return self.augmentation(sample)


def _compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    n_pos = float((labels == 1).sum().item())
    n_neg = float((labels == 0).sum().item())
    if n_pos < 1:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


def make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    labels = labels.cpu()
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()

    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)

    weights = torch.where(labels == 1, torch.tensor(w_pos), torch.tensor(w_neg)).double()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class SyntheticTrainer:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("=" * 70)
        logger.info("PHASE 1: SYNTHETIC DATA PRE-TRAINING")
        logger.info("=" * 70)
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Device: {self.device}")

        self.model = self._build_model()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["phase1"]["learning_rate"],
            weight_decay=self.config["training"]["phase1"].get("weight_decay", 1e-5),
        )

        self.use_amp = bool(self.config["training"].get("mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.train_metrics = PixelMetrics(threshold=self.config["evaluation"]["default_threshold"])
        self.val_metrics = PixelMetrics(threshold=self.config["evaluation"]["default_threshold"])

        self.early_stopping = EarlyStopping(
            patience=self.config["training"]["early_stopping"]["patience"],
            min_delta=self.config["training"]["early_stopping"]["min_delta"],
            mode="min",
        )

        checkpoint_dir = Path(self.config["paths"]["experiments"]) / "phase1_synthetic"
        self.checkpoint = ModelCheckpoint(str(checkpoint_dir), mode="min", save_best_only=True)

        self.scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type=self.config["training"]["scheduler"]["type"],
            factor=self.config["training"]["scheduler"]["factor"],
            patience=self.config["training"]["scheduler"]["patience"],
            min_lr=self.config["training"]["scheduler"]["min_lr"],
        )

        logger.info("✓ Trainer initialized")
        logger.info("=" * 70)

        self.criterion = None

    def _build_model(self):
        cfg = self.config["model"]
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
        return model.to(self.device)
    
    def _make_augmentation(self):
        """Create augmentation pipeline from config"""
        aug_cfg = self.config["training"]["phase1"].get("augmentation", {})
        
        if not aug_cfg.get("enabled", False):
            logger.info("Augmentation: DISABLED")
            return NoAugmentation()
        
        logger.info("Augmentation: ENABLED")
        logger.info(f"  Apply prob: {aug_cfg.get('apply_prob', 0.5)}")
        
        return PixelAugmentation(
            spectral_jitter_std=aug_cfg.get("spectral_jitter_std", 0.02),
            band_dropout_prob=aug_cfg.get("band_dropout_prob", 0.05),
            spatial_rotation=aug_cfg.get("spatial_rotation", True),
            spatial_flip=aug_cfg.get("spatial_flip", True),
            gaussian_noise_std=aug_cfg.get("gaussian_noise_std", 0.01),
            apply_prob=aug_cfg.get("apply_prob", 0.5),
        )

    def _make_loader(self, dataset: PreprocessedDataset, batch_size: int, shuffle: bool, balance: bool):
        num_workers = int(self.config["hardware"]["num_workers"])
        pin_memory = bool(self.config["hardware"]["pin_memory"])

        if balance:
            sampler = make_weighted_sampler(dataset.labels)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def train_epoch(self, loader):
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", ncols=100):
            spectrum = batch["spectrum"].to(self.device)
            spatial = batch["spatial"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(spectrum, spatial)
                    loss, _ = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(spectrum, spatial)
                loss, _ = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            probs = torch.sigmoid(logits)
            self.train_metrics.update(probs, labels)
            total_loss += float(loss.item())

        m = self.train_metrics.compute()
        m["loss"] = total_loss / max(len(loader), 1)
        return m

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Validation", ncols=100):
            spectrum = batch["spectrum"].to(self.device)
            spatial = batch["spatial"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(spectrum, spatial)
                    loss, _ = self.criterion(logits, labels)
            else:
                logits = self.model(spectrum, spatial)
                loss, _ = self.criterion(logits, labels)

            probs = torch.sigmoid(logits)
            self.val_metrics.update(probs, labels)
            total_loss += float(loss.item())

        m = self.val_metrics.compute()
        m["loss"] = total_loss / max(len(loader), 1)
        return m

    def train(self, train_data: str, val_data: str, use_preprocessed: bool = True):
        if not use_preprocessed:
            raise ValueError("Only preprocessed .pt files supported")

        # Create augmentation
        train_aug = self._make_augmentation()
        val_aug = NoAugmentation()  # No augmentation for validation

        train_ds = PreprocessedDataset(train_data, augmentation=train_aug)
        val_ds = PreprocessedDataset(val_data, augmentation=val_aug)

        pos_weight = _compute_pos_weight(train_ds.labels).to(self.device)
        logger.info(f"Phase1 pos_weight (neg/pos): {pos_weight.item():.3f}")
        self.criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=0.5, pos_weight=pos_weight)

        train_loader = self._make_loader(train_ds, self.config["training"]["phase1"]["batch_size"], shuffle=True, balance=True)
        val_loader = self._make_loader(val_ds, self.config["training"]["phase1"]["batch_size"], shuffle=False, balance=False)

        logger.info(f"Train: {len(train_ds):,} samples, {len(train_loader)} batches")
        logger.info(f"Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

        best_val = float("inf")
        epochs = int(self.config["training"]["phase1"]["epochs"])

        for epoch in range(epochs):
            logger.info(f"\nEPOCH {epoch+1}/{epochs}\n" + "-" * 70)

            tr = self.train_epoch(train_loader)
            logger.info(f"[TRAIN] Loss: {tr['loss']:.4f} | F1: {tr['f1']:.4f} | P: {tr['precision']:.4f} | R: {tr['recall']:.4f}")

            va = self.validate(val_loader)
            logger.info(f"[VAL]   Loss: {va['loss']:.4f} | F1: {va['f1']:.4f} | P: {va['precision']:.4f} | R: {va['recall']:.4f}")

            self.scheduler.step(va["loss"])
            self.checkpoint(self.model, va["loss"], epoch, self.optimizer, {"train": tr, "val": va})

            if va["loss"] < best_val:
                best_val = va["loss"]
                logger.info(f"★ NEW BEST: {best_val:.4f}")

            if self.early_stopping(va["loss"]):
                logger.info("Early stopping triggered")
                break

        logger.info("=" * 70)
        logger.info(f"✓ TRAINING COMPLETE! Best val loss: {best_val:.4f}")
        logger.info("=" * 70)
