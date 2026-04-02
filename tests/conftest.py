"""
Pytest Configuration and Fixtures
Shared fixtures for all tests
"""

import pytest
import numpy as np
import torch
import h5py
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_hsi_cube():
    """Generate sample HSI cube for testing"""
    # (H, W, C) = (256, 256, 204)
    cube = np.random.rand(256, 256, 204).astype(np.float32)
    return cube


@pytest.fixture
def small_hsi_cube():
    """Generate small HSI cube for faster tests"""
    # (H, W, C) = (32, 32, 204)
    cube = np.random.rand(32, 32, 204).astype(np.float32)
    return cube


@pytest.fixture
def sample_mask():
    """Generate sample binary mask"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Add some FM regions
    mask[100:150, 100:150] = 1
    mask[200:220, 200:220] = 1
    return mask


@pytest.fixture
def small_mask():
    """Generate small binary mask"""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    return mask


@pytest.fixture
def sample_spectrum():
    """Generate sample spectrum"""
    spectrum = np.random.rand(204).astype(np.float32)
    return spectrum


@pytest.fixture
def sample_wavelengths():
    """Generate wavelength array"""
    return np.linspace(400, 1000, 204)


@pytest.fixture
def sample_hdf5(temp_dir, sample_hsi_cube, sample_mask):
    """Create sample HDF5 file"""
    hdf5_path = temp_dir / "test_data.h5"
    
    with h5py.File(hdf5_path, 'w') as hf:
        for i in range(5):  # 5 samples
            grp = hf.create_group(f"sample_{i:04d}")
            grp.create_dataset('hsi_cube', data=sample_hsi_cube)
            grp.create_dataset('mask', data=sample_mask)
            grp.attrs['sample_id'] = f"sample_{i:04d}"
    
    yield hdf5_path
    
    # Cleanup
    if hdf5_path.exists():
        hdf5_path.unlink()


@pytest.fixture
def sample_fm_signatures():
    """Generate sample FM signatures"""
    n_bands = 204
    fm_classes = ['plastic', 'textile', 'metal', 'paper']
    
    signatures = {}
    for fm_class in fm_classes:
        signatures[fm_class] = {
            'mean': np.random.rand(n_bands).astype(np.float32),
            'std': np.random.rand(n_bands).astype(np.float32) * 0.1,
            'median': np.random.rand(n_bands).astype(np.float32),
            'p25': np.random.rand(n_bands).astype(np.float32),
            'p75': np.random.rand(n_bands).astype(np.float32),
            'n_samples': 10,
            'n_pixels': 1000
        }
    
    return signatures


@pytest.fixture
def device():
    """Get available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_config():
    """Generate sample configuration"""
    return {
        'preprocessing': {
            'target_bands': 204,
            'target_resolution': [256, 256],
            'spectral_range': [400, 1000],
            'normalization': {'method': 'l2_per_pixel'},
            'interpolation': 'linear',
            'batch_size': 10
        },
        'model': {
            'input_bands': 204,
            'spectral_branch': {
                'architecture': 'Conv1D',
                'channels': [32, 64]
            },
            'spatial_branch': {
                'input_channels': 3,
                'channels': [16, 32]
            },
            'fusion': {
                'hidden_dim': 64,
                'dropout': 0.3
            }
        },
        'synthetic': {
            'n_synthetic_per_clean': 10,
            'fm_size_range': [2, 5],
            'n_objects_per_cube': [3, 7],
            'gmm_aware': True
        }
    }
