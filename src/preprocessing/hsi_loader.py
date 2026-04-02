"""
HSI Data Loader
Handles loading of various HSI formats with caching
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class HSILoader:
    """
    Unified loader for HSI data from different sources
    """
    
    def __init__(self, cache_size: int = 10):
        """
        Args:
            cache_size: Number of cubes to keep in memory
        """
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
    
    def load_from_hdf5(self, hdf5_path: str, 
                       sample_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Load HSI cube from HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            sample_idx: Sample index
            
        Returns:
            cube: HSI data (H, W, C)
            metadata: Dictionary with sample information
        """
        cache_key = f"{hdf5_path}_{sample_idx}"
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Loading from cache: {cache_key}")
            return self.cache[cache_key]
        
        with h5py.File(hdf5_path, 'r') as hf:
            sample_key = f"sample_{sample_idx:04d}"
            
            if sample_key not in hf:
                raise KeyError(f"{sample_key} not found in {hdf5_path}")
            
            grp = hf[sample_key]
            cube = grp['hsi_cube'][:]
            
            # Load metadata
            metadata = dict(grp.attrs)
            
            # Additional data if available
            if 'mask' in grp:
                metadata['mask'] = grp['mask'][:]
        
        # Update cache
        self._update_cache(cache_key, (cube, metadata))
        
        return cube, metadata
    
    def load_batch_from_hdf5(self, hdf5_path: str, 
                            sample_indices: list) -> list:
        """
        Load multiple samples efficiently
        
        Args:
            hdf5_path: Path to HDF5 file
            sample_indices: List of sample indices
            
        Returns:
            samples: List of (cube, metadata) tuples
        """
        samples = []
        
        with h5py.File(hdf5_path, 'r') as hf:
            for idx in sample_indices:
                sample_key = f"sample_{idx:04d}"
                
                if sample_key in hf:
                    grp = hf[sample_key]
                    cube = grp['hsi_cube'][:]
                    metadata = dict(grp.attrs)
                    
                    if 'mask' in grp:
                        metadata['mask'] = grp['mask'][:]
                    
                    samples.append((cube, metadata))
        
        return samples
    
    def get_hdf5_info(self, hdf5_path: str) -> Dict:
        """
        Get information about HDF5 dataset
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            info: Dictionary with dataset information
        """
        with h5py.File(hdf5_path, 'r') as hf:
            n_samples = len([k for k in hf.keys() if k.startswith('sample_')])
            
            # Get sample properties from first sample
            if n_samples > 0:
                first_sample = hf['sample_0000']
                cube_shape = first_sample['hsi_cube'].shape
                cube_dtype = first_sample['hsi_cube'].dtype
            else:
                cube_shape = None
                cube_dtype = None
            
            info = {
                'n_samples': n_samples,
                'cube_shape': cube_shape,
                'cube_dtype': str(cube_dtype),
                'file_size_mb': Path(hdf5_path).stat().st_size / 1e6
            }
        
        return info
    
    def _update_cache(self, key: str, value: Tuple):
        """Update LRU cache"""
        if key in self.cache:
            self.cache_order.remove(key)
        
        self.cache[key] = value
        self.cache_order.append(key)
        
        # Remove oldest if cache is full
        while len(self.cache) > self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_order.clear()
        logger.info("Cache cleared")
