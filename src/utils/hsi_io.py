"""
HSI I/O Utilities
Read and write hyperspectral data
"""

import numpy as np
import h5py
import spectral as spy
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HSIReader:
    """
    Read hyperspectral data from various formats
    """
    
    @staticmethod
    def read_envi(hdr_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Read ENVI format HSI
        
        Args:
            hdr_path: Path to .hdr file
            
        Returns:
            cube: HSI cube (H, W, C)
            metadata: Metadata dictionary
        """
        img = spy.open_image(hdr_path)
        cube = img.load()
        
        metadata = {
            'wavelengths': img.metadata.get('wavelength', None),
            'bands': img.nbands,
            'lines': img.nrows,
            'samples': img.ncols,
            'interleave': img.metadata.get('interleave', 'unknown')
        }
        
        return cube, metadata
    
    @staticmethod
    def read_hdf5(hdf5_path: str, sample_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Read from HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            sample_idx: Sample index
            
        Returns:
            cube: HSI cube
            metadata: Metadata
        """
        with h5py.File(hdf5_path, 'r') as hf:
            sample_key = f"sample_{sample_idx:04d}"
            
            if sample_key not in hf:
                raise KeyError(f"{sample_key} not found")
            
            grp = hf[sample_key]
            cube = grp['hsi_cube'][:]
            metadata = dict(grp.attrs)
            
            if 'mask' in grp:
                metadata['mask'] = grp['mask'][:]
        
        return cube, metadata
    
    @staticmethod
    def read_npy(npy_path: str) -> np.ndarray:
        """Read from numpy file"""
        return np.load(npy_path)
    
    @staticmethod
    def read_npz(npz_path: str) -> Dict:
        """Read from compressed numpy file"""
        return dict(np.load(npz_path))


class HSIWriter:
    """
    Write hyperspectral data
    """
    
    @staticmethod
    def write_hdf5(output_path: str, cube: np.ndarray, 
                   metadata: Optional[Dict] = None,
                   compression: bool = True):
        """
        Write to HDF5
        
        Args:
            output_path: Output path
            cube: HSI cube
            metadata: Optional metadata
            compression: Whether to compress
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as hf:
            if compression:
                hf.create_dataset(
                    'hsi_cube',
                    data=cube,
                    compression='gzip',
                    compression_opts=4
                )
            else:
                hf.create_dataset('hsi_cube', data=cube)
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        hf.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        hf.create_dataset(key, data=value)
        
        logger.info(f"Saved HSI cube to {output_path}")
    
    @staticmethod
    def write_npy(output_path: str, cube: np.ndarray):
        """Write to numpy file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, cube)
        logger.info(f"Saved to {output_path}")
    
    @staticmethod
    def write_npz(output_path: str, **data):
        """Write to compressed numpy file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **data)
        logger.info(f"Saved to {output_path}")
