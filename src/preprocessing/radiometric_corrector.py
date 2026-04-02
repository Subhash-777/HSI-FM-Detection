"""
Radiometric Correction
Normalize reflectance values and apply corrections
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RadiometricCorrector:
    """
    Apply radiometric corrections to HSI data
    """
    
    def __init__(self, method: str = 'l2_per_pixel'):
        """
        Args:
            method: Normalization method
                   - 'l2_per_pixel': L2 norm per pixel spectrum
                   - 'minmax': Min-max normalization
                   - 'zscore': Z-score normalization
                   - 'percentile': Percentile-based clipping
        """
        self.method = method
        logger.info(f"RadiometricCorrector initialized: method={method}")
    
    def normalize_cube(self, cube: np.ndarray, 
                      clip_percentile: Optional[float] = None) -> np.ndarray:
        """
        Normalize HSI cube
        
        Args:
            cube: Input HSI cube (H, W, C)
            clip_percentile: Percentile for clipping outliers (e.g., 99.5)
            
        Returns:
            normalized_cube: Normalized HSI cube
        """
        logger.info(f"Normalizing cube with method: {self.method}")
        
        # Clip outliers if requested
        if clip_percentile is not None:
            cube = self._clip_outliers(cube, clip_percentile)
        
        if self.method == 'l2_per_pixel':
            normalized = self._l2_normalize(cube)
        elif self.method == 'minmax':
            normalized = self._minmax_normalize(cube)
        elif self.method == 'zscore':
            normalized = self._zscore_normalize(cube)
        elif self.method == 'percentile':
            normalized = self._percentile_normalize(cube)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return normalized
    
    def _l2_normalize(self, cube: np.ndarray) -> np.ndarray:
        """
        L2 normalization per pixel spectrum
        Each pixel's spectrum is divided by its L2 norm
        """
        H, W, C = cube.shape
        normalized = np.zeros_like(cube, dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                spectrum = cube[i, j, :]
                norm = np.linalg.norm(spectrum)
                if norm > 0:
                    normalized[i, j, :] = spectrum / norm
                else:
                    normalized[i, j, :] = spectrum
        
        return normalized
    
    def _minmax_normalize(self, cube: np.ndarray) -> np.ndarray:
        """
        Min-max normalization to [0, 1] range
        """
        min_val = cube.min()
        max_val = cube.max()
        
        if max_val - min_val > 0:
            normalized = (cube - min_val) / (max_val - min_val)
        else:
            normalized = cube
        
        return normalized.astype(np.float32)
    
    def _zscore_normalize(self, cube: np.ndarray) -> np.ndarray:
        """
        Z-score normalization (zero mean, unit variance)
        """
        mean = cube.mean()
        std = cube.std()
        
        if std > 0:
            normalized = (cube - mean) / std
        else:
            normalized = cube - mean
        
        return normalized.astype(np.float32)
    
    def _percentile_normalize(self, cube: np.ndarray, 
                             lower: float = 1, upper: float = 99) -> np.ndarray:
        """
        Percentile-based normalization
        """
        p_lower = np.percentile(cube, lower)
        p_upper = np.percentile(cube, upper)
        
        normalized = np.clip(cube, p_lower, p_upper)
        normalized = (normalized - p_lower) / (p_upper - p_lower)
        
        return normalized.astype(np.float32)
    
    def _clip_outliers(self, cube: np.ndarray, percentile: float) -> np.ndarray:
        """
        Clip outliers beyond specified percentile
        """
        upper_limit = np.percentile(cube, percentile)
        lower_limit = np.percentile(cube, 100 - percentile)
        
        clipped = np.clip(cube, lower_limit, upper_limit)
        
        logger.debug(f"Clipped outliers: [{lower_limit:.4f}, {upper_limit:.4f}]")
        
        return clipped
    
    def correct_dark_current(self, cube: np.ndarray, 
                            dark_reference: np.ndarray) -> np.ndarray:
        """
        Correct for dark current using dark reference
        
        Args:
            cube: Input HSI cube
            dark_reference: Dark reference measurement
            
        Returns:
            corrected_cube: Dark-corrected cube
        """
        corrected = cube - dark_reference
        corrected = np.maximum(corrected, 0)  # Ensure non-negative
        
        return corrected
    
    def correct_white_reference(self, cube: np.ndarray,
                               white_reference: np.ndarray) -> np.ndarray:
        """
        Correct using white reference for reflectance calibration
        
        Args:
            cube: Input HSI cube
            white_reference: White reference measurement
            
        Returns:
            corrected_cube: Reflectance-calibrated cube
        """
        # Avoid division by zero
        white_reference = np.maximum(white_reference, 1e-6)
        
        corrected = cube / white_reference
        corrected = np.clip(corrected, 0, 1)  # Reflectance in [0, 1]
        
        return corrected
