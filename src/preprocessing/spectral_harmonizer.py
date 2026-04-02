"""
Spectral Harmonization
Harmonize all datasets to common spectral configuration (204 bands, 400-1000nm)
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SpectralHarmonizer:
    """
    Harmonize hyperspectral data to common spectral configuration
    - Target: 204 bands @ 400-1000nm
    - Methods: Linear interpolation or PCA-based reduction
    """
    
    def __init__(self, 
                 target_bands: int = 204,
                 target_range: Tuple[float, float] = (400, 1000),
                 method: str = 'linear'):
        """
        Args:
            target_bands: Target number of spectral bands
            target_range: Target wavelength range (min_nm, max_nm)
            method: Interpolation method ('linear', 'cubic', 'pca')
        """
        self.target_bands = target_bands
        self.target_range = target_range
        self.method = method
        
        # Generate target wavelengths
        self.target_wavelengths = np.linspace(
            target_range[0], 
            target_range[1], 
            target_bands
        )
        
        logger.info(f"SpectralHarmonizer initialized: {target_bands} bands, "
                   f"{target_range[0]}-{target_range[1]} nm, method={method}")
    
    def harmonize_cube(self, 
                       cube: np.ndarray,
                       source_wavelengths: Optional[np.ndarray] = None,
                       source_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Harmonize HSI cube to target spectral configuration
        
        Args:
            cube: Input HSI cube (H, W, C)
            source_wavelengths: Source wavelength array (optional)
            source_range: Source wavelength range (min_nm, max_nm)
            
        Returns:
            harmonized_cube: Output HSI cube (H, W, target_bands)
        """
        H, W, C = cube.shape
        
        # Generate source wavelengths if not provided
        if source_wavelengths is None:
            if source_range is None:
                source_range = self.target_range
            source_wavelengths = np.linspace(source_range[0], source_range[1], C)
        
        # Check if already harmonized
        if C == self.target_bands and np.allclose(source_wavelengths, self.target_wavelengths):
            logger.debug("Cube already harmonized, returning as-is")
            return cube
        
        logger.info(f"Harmonizing cube: {C} bands → {self.target_bands} bands")
        
        # Downsample or upsample
        if self.method in ['linear', 'cubic']:
            harmonized = self._interpolate_cube(cube, source_wavelengths)
        else:
            raise ValueError(f"Unknown harmonization method: {self.method}")
        
        return harmonized
    
    def _interpolate_cube(self, cube: np.ndarray, 
                         source_wavelengths: np.ndarray) -> np.ndarray:
        """
        Interpolate cube to target wavelengths
        Uses row-wise processing to manage memory
        """
        H, W, C = cube.shape
        harmonized = np.zeros((H, W, self.target_bands), dtype=np.float32)
        
        # Process row by row to save memory
        for i in range(H):
            for j in range(W):
                spectrum = cube[i, j, :]
                
                # Interpolate
                f = interp1d(
                    source_wavelengths, 
                    spectrum, 
                    kind=self.method,
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                harmonized[i, j, :] = f(self.target_wavelengths)
        
        return harmonized
    
    def harmonize_agrifood_300_to_204(self, cube_300: np.ndarray) -> np.ndarray:
        """
        Specialized method for AgriFoodAnomaly: 300 → 204 bands
        
        Args:
            cube_300: Input cube with 300 bands (H, W, 300)
            
        Returns:
            cube_204: Output cube with 204 bands (H, W, 204)
        """
        source_wavelengths = np.linspace(400, 1000, 300)
        return self.harmonize_cube(cube_300, source_wavelengths)
    
    def harmonize_batch(self, cubes: list, 
                       source_wavelengths: Optional[np.ndarray] = None) -> list:
        """
        Harmonize a batch of cubes
        
        Args:
            cubes: List of HSI cubes
            source_wavelengths: Source wavelength array
            
        Returns:
            harmonized_cubes: List of harmonized cubes
        """
        from tqdm import tqdm
        
        harmonized = []
        for cube in tqdm(cubes, desc="Harmonizing batch"):
            harmonized.append(
                self.harmonize_cube(cube, source_wavelengths)
            )
        
        return harmonized
    
    def validate_harmonization(self, original_cube: np.ndarray,
                              harmonized_cube: np.ndarray) -> Dict:
        """
        Validate harmonization quality
        
        Args:
            original_cube: Original HSI cube
            harmonized_cube: Harmonized HSI cube
            
        Returns:
            metrics: Dictionary with validation metrics
        """
        # Sample random pixels for comparison
        H, W, _ = original_cube.shape
        n_samples = min(100, H * W)
        
        indices = np.random.choice(H * W, n_samples, replace=False)
        rows = indices // W
        cols = indices % W
        
        # Compare spectral profiles (simplified - just check shape preservation)
        metrics = {
            'original_shape': original_cube.shape,
            'harmonized_shape': harmonized_cube.shape,
            'target_bands_match': harmonized_cube.shape[2] == self.target_bands,
            'min_value': float(harmonized_cube.min()),
            'max_value': float(harmonized_cube.max()),
            'mean_value': float(harmonized_cube.mean()),
            'std_value': float(harmonized_cube.std())
        }
        
        return metrics
