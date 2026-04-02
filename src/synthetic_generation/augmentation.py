"""
HSI Data Augmentation
Augmentations specific to hyperspectral imaging
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HSIAugmenter:
    """
    Hyperspectral data augmentation
    """
    
    def __init__(self,
                 spectral_jitter: float = 0.05,
                 band_dropout_rate: float = 0.1,
                 gaussian_noise: float = 0.02,
                 spatial_rotation: bool = True):
        """
        Args:
            spectral_jitter: Amount of spectral jitter (0-1)
            band_dropout_rate: Probability of dropping spectral bands
            gaussian_noise: Gaussian noise std
            spatial_rotation: Whether to apply spatial rotations
        """
        self.spectral_jitter = spectral_jitter
        self.band_dropout_rate = band_dropout_rate
        self.gaussian_noise = gaussian_noise
        self.spatial_rotation = spatial_rotation
    
    def augment(self, cube: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """
        Apply random augmentations
        
        Args:
            cube: HSI cube (H, W, C)
            mask: Binary mask (H, W)
            
        Returns:
            aug_cube: Augmented cube
            aug_mask: Augmented mask
        """
        aug_cube = cube.copy()
        aug_mask = mask.copy() if mask is not None else None
        
        # Spectral augmentations
        if np.random.rand() < 0.5:
            aug_cube = self.add_spectral_jitter(aug_cube)
        
        if np.random.rand() < 0.3:
            aug_cube = self.band_dropout(aug_cube)
        
        if np.random.rand() < 0.5:
            aug_cube = self.add_gaussian_noise(aug_cube)
        
        # Spatial augmentations
        if self.spatial_rotation and np.random.rand() < 0.5:
            aug_cube, aug_mask = self.rotate(aug_cube, aug_mask)
        
        if np.random.rand() < 0.5:
            aug_cube, aug_mask = self.flip(aug_cube, aug_mask)
        
        return aug_cube, aug_mask
    
    def add_spectral_jitter(self, cube: np.ndarray) -> np.ndarray:
        """Add spectral jitter"""
        H, W, C = cube.shape
        
        # Per-pixel spectral jitter
        jitter = np.random.normal(1.0, self.spectral_jitter, (H, W, C))
        jittered = cube * jitter
        jittered = np.clip(jittered, 0, 1)
        
        return jittered.astype(cube.dtype)
    
    def band_dropout(self, cube: np.ndarray) -> np.ndarray:
        """Randomly dropout spectral bands"""
        H, W, C = cube.shape
        
        # Select bands to drop
        n_drop = int(C * self.band_dropout_rate)
        if n_drop == 0:
            return cube
        
        drop_indices = np.random.choice(C, n_drop, replace=False)
        
        dropped = cube.copy()
        dropped[:, :, drop_indices] = 0
        
        return dropped
    
    def add_gaussian_noise(self, cube: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.gaussian_noise, cube.shape)
        noisy = cube + noise
        noisy = np.clip(noisy, 0, 1)
        
        return noisy.astype(cube.dtype)
    
    def rotate(self, cube: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Random 90-degree rotation"""
        k = np.random.randint(0, 4)  # 0, 90, 180, 270 degrees
        
        rotated_cube = np.rot90(cube, k=k, axes=(0, 1))
        rotated_mask = np.rot90(mask, k=k) if mask is not None else None
        
        return rotated_cube, rotated_mask
    
    def flip(self, cube: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """Random flip"""
        axis = np.random.randint(0, 2)  # 0: vertical, 1: horizontal
        
        flipped_cube = np.flip(cube, axis=axis)
        flipped_mask = np.flip(mask, axis=axis) if mask is not None else None
        
        return flipped_cube, flipped_mask
