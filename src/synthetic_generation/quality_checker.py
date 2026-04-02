"""
Synthetic Quality Checker
Validate quality and realism of synthetic data
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SyntheticQualityChecker:
    """
    Check quality of synthetic samples
    """
    
    def __init__(self):
        self.checks = []
    
    def check_sample(self, synthetic_cube: np.ndarray,
                    clean_cube: np.ndarray,
                    mask: np.ndarray) -> Dict:
        """
        Perform quality checks on synthetic sample
        
        Args:
            synthetic_cube: Synthetic cube
            clean_cube: Original clean cube
            mask: FM mask
            
        Returns:
            results: Dictionary with check results
        """
        results = {}
        
        # 1. Check value range
        results['value_range_ok'] = self._check_value_range(synthetic_cube)
        
        # 2. Check FM region statistics
        results['fm_stats'] = self._check_fm_region(synthetic_cube, mask)
        
        # 3. Check preservation of clean regions
        results['clean_preservation'] = self._check_clean_preservation(
            synthetic_cube, clean_cube, mask
        )
        
        # 4. Check mask properties
        results['mask_properties'] = self._check_mask(mask)
        
        # Overall quality score
        results['quality_score'] = self._compute_quality_score(results)
        
        return results
    
    def _check_value_range(self, cube: np.ndarray) -> bool:
        """Check if values are in valid range"""
        return (cube.min() >= 0) and (cube.max() <= 1)
    
    def _check_fm_region(self, cube: np.ndarray, mask: np.ndarray) -> Dict:
        """Check statistics of FM region"""
        if mask.sum() == 0:
            return {'valid': False}
        
        fm_pixels = cube[mask > 0]
        
        return {
            'valid': True,
            'mean': float(fm_pixels.mean()),
            'std': float(fm_pixels.std()),
            'n_pixels': int(mask.sum())
        }
    
    def _check_clean_preservation(self, synthetic_cube: np.ndarray,
                                  clean_cube: np.ndarray,
                                  mask: np.ndarray) -> Dict:
        """Check how well clean regions are preserved"""
        clean_mask = (mask == 0)
        
        if clean_mask.sum() == 0:
            return {'valid': False}
        
        synthetic_clean = synthetic_cube[clean_mask]
        original_clean = clean_cube[clean_mask]
        
        # Compute difference
        diff = np.abs(synthetic_clean - original_clean)
        
        return {
            'valid': True,
            'mean_diff': float(diff.mean()),
            'max_diff': float(diff.max()),
            'preservation_ratio': float(1 - diff.mean())
        }
    
    def _check_mask(self, mask: np.ndarray) -> Dict:
        """Check mask properties"""
        return {
            'n_pixels': int(mask.sum()),
            'coverage': float(mask.sum() / mask.size),
            'n_components': self._count_connected_components(mask)
        }
    
    def _count_connected_components(self, mask: np.ndarray) -> int:
        """Count number of connected components in mask"""
        import cv2
        n_components, _ = cv2.connectedComponents(mask.astype(np.uint8))
        return n_components - 1  # Exclude background
    
    def _compute_quality_score(self, results: Dict) -> float:
        """Compute overall quality score (0-1)"""
        score = 0.0
        
        # Value range (0.2)
        if results['value_range_ok']:
            score += 0.2
        
        # FM region validity (0.3)
        if results['fm_stats']['valid']:
            score += 0.3
        
        # Clean preservation (0.5)
        if results['clean_preservation']['valid']:
            pres_ratio = results['clean_preservation']['preservation_ratio']
            score += 0.5 * pres_ratio
        
        return score
