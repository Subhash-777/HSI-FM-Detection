"""
FM Insertion Strategies
Different methods for inserting FMs into clean food cubes
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class InsertionStrategy(ABC):
    """Base class for FM insertion strategies"""
    
    @abstractmethod
    def insert(self, cube: np.ndarray, fm_spectrum: np.ndarray,
              center: Tuple[int, int], size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert FM into cube
        
        Args:
            cube: Clean HSI cube (H, W, C)
            fm_spectrum: FM spectral signature (C,)
            center: Center position (x, y)
            size: FM object size in pixels
            
        Returns:
            modified_cube: Cube with FM inserted
            mask: Binary mask of FM location
        """
        pass


class GaussianInsertion(InsertionStrategy):
    """
    Gaussian-based FM insertion with smooth edges
    """
    
    def __init__(self, alpha_range: Tuple[float, float] = (0.6, 0.9)):
        """
        Args:
            alpha_range: Range for alpha blending strength
        """
        self.alpha_range = alpha_range
    
    def insert(self, cube: np.ndarray, fm_spectrum: np.ndarray,
              center: Tuple[int, int], size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Insert FM with Gaussian falloff"""
        H, W, C = cube.shape
        cx, cy = center
        
        modified_cube = cube.copy()
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # Random alpha
        base_alpha = np.random.uniform(*self.alpha_range)
        
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                x, y = cx + dx, cy + dy
                
                # Check bounds
                if not (0 <= x < H and 0 <= y < W):
                    continue
                
                # Gaussian falloff based on distance
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist <= size:
                    # Gaussian weight
                    sigma = size / 2
                    weight = np.exp(-(dist**2) / (2 * sigma**2))
                    
                    alpha = base_alpha * weight
                    
                    # Blend
                    modified_cube[x, y, :] = (
                        alpha * fm_spectrum + 
                        (1 - alpha) * cube[x, y, :]
                    )
                    
                    # Update mask (threshold at 50% alpha)
                    if alpha > 0.5:
                        mask[x, y] = 1
        
        return modified_cube, mask


class EdgeBlendingInsertion(InsertionStrategy):
    """
    Edge-blending insertion with distance-based alpha
    """
    
    def __init__(self, alpha_center: float = 0.85, 
                 edge_smoothness: float = 0.3):
        """
        Args:
            alpha_center: Alpha value at center
            edge_smoothness: Smoothness factor for edges
        """
        self.alpha_center = alpha_center
        self.edge_smoothness = edge_smoothness
    
    def insert(self, cube: np.ndarray, fm_spectrum: np.ndarray,
              center: Tuple[int, int], size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Insert FM with edge blending"""
        H, W, C = cube.shape
        cx, cy = center
        
        modified_cube = cube.copy()
        mask = np.zeros((H, W), dtype=np.uint8)
        
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                x, y = cx + dx, cy + dy
                
                if not (0 <= x < H and 0 <= y < W):
                    continue
                
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist <= size:
                    # Linear falloff with smoothness
                    alpha = self.alpha_center * (1 - (dist / size) * self.edge_smoothness)
                    alpha = np.clip(alpha, 0, 1)
                    
                    modified_cube[x, y, :] = (
                        alpha * fm_spectrum + 
                        (1 - alpha) * cube[x, y, :]
                    )
                    
                    if alpha > 0.3:
                        mask[x, y] = 1
        
        return modified_cube, mask


class RealisticInsertion(InsertionStrategy):
    """
    Realistic insertion with spectral variance and texture
    """
    
    def __init__(self, alpha_range: Tuple[float, float] = (0.6, 0.9),
                 spectral_noise: float = 0.05):
        """
        Args:
            alpha_range: Range for alpha blending
            spectral_noise: Amount of spectral noise to add
        """
        self.alpha_range = alpha_range
        self.spectral_noise = spectral_noise
    
    def insert(self, cube: np.ndarray, fm_spectrum: np.ndarray,
              center: Tuple[int, int], size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Insert FM with realistic variations"""
        H, W, C = cube.shape
        cx, cy = center
        
        modified_cube = cube.copy()
        mask = np.zeros((H, W), dtype=np.uint8)
        
        base_alpha = np.random.uniform(*self.alpha_range)
        
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                x, y = cx + dx, cy + dy
                
                if not (0 <= x < H and 0 <= y < W):
                    continue
                
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist <= size:
                    # Add spectral variance
                    noise = np.random.normal(0, self.spectral_noise, C)
                    noisy_spectrum = fm_spectrum + fm_spectrum * noise
                    noisy_spectrum = np.clip(noisy_spectrum, 0, 1)
                    
                    # Distance-based alpha
                    alpha = base_alpha * (1 - dist / (size + 1))
                    
                    # Add spatial texture (small random variations)
                    alpha += np.random.normal(0, 0.05)
                    alpha = np.clip(alpha, 0, 1)
                    
                    modified_cube[x, y, :] = (
                        alpha * noisy_spectrum + 
                        (1 - alpha) * cube[x, y, :]
                    )
                    
                    if alpha > 0.4:
                        mask[x, y] = 1
        
        return modified_cube, mask
