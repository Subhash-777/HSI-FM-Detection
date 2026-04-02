"""
Training Callbacks
"""

import torch
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = (score < (self.best_score - self.min_delta) if self.mode == 'min' 
                   else score > (self.best_score + self.min_delta))
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(self, checkpoint_dir: str, mode: str = 'min', save_best_only: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
    
    def __call__(self, model, score: float, epoch: int, optimizer=None, extra_state=None):
        is_best = False
        if self.best_score is None:
            is_best = True
            self.best_score = score
        else:
            is_best = (score < self.best_score if self.mode == 'min' else score > self.best_score)
            if is_best:
                self.best_score = score
        
        if is_best or not self.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score,
                'best_score': self.best_score
            }
            
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if extra_state:
                checkpoint.update(extra_state)
            
            path = self.checkpoint_dir / ('best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, path)
            logger.info(f"Saved {'best model' if is_best else 'checkpoint'}: score={score:.4f}")


class LearningRateScheduler:
    """Learning rate scheduler callback"""
    
    def __init__(self, optimizer, scheduler_type: str = 'ReduceLROnPlateau',
                 factor: float = 0.5, patience: int = 5, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
            )
        elif scheduler_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=min_lr)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def step(self, metric=None):
        if self.scheduler_type == 'ReduceLROnPlateau':
            if metric is None:
                raise ValueError("metric required for ReduceLROnPlateau")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
