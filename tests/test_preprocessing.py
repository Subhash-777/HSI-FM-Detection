"""
Test Preprocessing Module
Tests for spectral harmonization, spatial normalization, and radiometric correction
"""

import pytest
import numpy as np
from pathlib import Path

from src.preprocessing.spectral_harmonizer import SpectralHarmonizer
from src.preprocessing.spatial_normalizer import SpatialNormalizer
from src.preprocessing.radiometric_corrector import RadiometricCorrector
from src.preprocessing.hsi_loader import HSILoader
from src.preprocessing.batch_processor import BatchProcessor


class TestSpectralHarmonizer:
    """Test SpectralHarmonizer class"""
    
    def test_initialization(self):
        """Test harmonizer initialization"""
        harmonizer = SpectralHarmonizer(
            target_bands=204,
            target_range=(400, 1000),
            method='linear'
        )
        
        assert harmonizer.target_bands == 204
        assert harmonizer.target_range == (400, 1000)
        assert len(harmonizer.target_wavelengths) == 204
    
    def test_harmonize_same_bands(self, small_hsi_cube):
        """Test harmonization when already at target bands"""
        harmonizer = SpectralHarmonizer(target_bands=204)
        
        # Cube already has 204 bands
        result = harmonizer.harmonize_cube(small_hsi_cube)
        
        assert result.shape == small_hsi_cube.shape
        np.testing.assert_array_almost_equal(result, small_hsi_cube)
    
    def test_harmonize_downsample(self):
        """Test downsampling from 300 to 204 bands"""
        harmonizer = SpectralHarmonizer(target_bands=204)
        
        # Create cube with 300 bands
        cube_300 = np.random.rand(32, 32, 300).astype(np.float32)
        source_wavelengths = np.linspace(400, 1000, 300)
        
        result = harmonizer.harmonize_cube(cube_300, source_wavelengths)
        
        assert result.shape == (32, 32, 204)
        assert result.dtype == np.float32
    
    def test_agrifood_harmonization(self):
        """Test specific AgriFoodAnomaly harmonization"""
        harmonizer = SpectralHarmonizer(target_bands=204)
        
        cube_300 = np.random.rand(32, 32, 300).astype(np.float32)
        result = harmonizer.harmonize_agrifood_300_to_204(cube_300)
        
        assert result.shape == (32, 32, 204)
    
    def test_validate_harmonization(self, small_hsi_cube):
        """Test harmonization validation"""
        harmonizer = SpectralHarmonizer(target_bands=204)
        
        # Create slightly different cube
        cube_300 = np.random.rand(32, 32, 300).astype(np.float32)
        harmonized = harmonizer.harmonize_agrifood_300_to_204(cube_300)
        
        metrics = harmonizer.validate_harmonization(cube_300, harmonized)
        
        assert 'original_shape' in metrics
        assert 'harmonized_shape' in metrics
        assert metrics['target_bands_match'] == True


class TestSpatialNormalizer:
    """Test SpatialNormalizer class"""
    
    def test_initialization(self):
        """Test normalizer initialization"""
        normalizer = SpatialNormalizer(target_size=(256, 256))
        assert normalizer.target_size == (256, 256)
    
    def test_resize_cube(self):
        """Test cube resizing"""
        normalizer = SpatialNormalizer(target_size=(128, 128))
        
        # Create 256x256 cube
        cube = np.random.rand(256, 256, 204).astype(np.float32)
        result = normalizer.resize_cube(cube)
        
        assert result.shape == (128, 128, 204)
        assert result.dtype == np.float32
    
    def test_resize_already_correct_size(self, small_hsi_cube):
        """Test when cube is already at target size"""
        normalizer = SpatialNormalizer(target_size=(32, 32))
        
        result = normalizer.resize_cube(small_hsi_cube)
        
        assert result.shape == small_hsi_cube.shape
        np.testing.assert_array_equal(result, small_hsi_cube)
    
    def test_resize_mask(self, sample_mask):
        """Test mask resizing"""
        normalizer = SpatialNormalizer(target_size=(128, 128))
        
        result = normalizer.resize_mask(sample_mask)
        
        assert result.shape == (128, 128)
        assert result.dtype == bool
    
    def test_crop_center(self, sample_hsi_cube):
        """Test center cropping"""
        normalizer = SpatialNormalizer(target_size=(128, 128))
        
        result = normalizer.crop_center(sample_hsi_cube, crop_size=(128, 128))
        
        assert result.shape == (128, 128, 204)
    
    def test_crop_too_small(self):
        """Test cropping when image is too small"""
        normalizer = SpatialNormalizer()
        cube = np.random.rand(64, 64, 204).astype(np.float32)
        
        with pytest.raises(ValueError):
            normalizer.crop_center(cube, crop_size=(128, 128))


class TestRadiometricCorrector:
    """Test RadiometricCorrector class"""
    
    def test_initialization(self):
        """Test corrector initialization"""
        corrector = RadiometricCorrector(method='l2_per_pixel')
        assert corrector.method == 'l2_per_pixel'
    
    def test_l2_normalization(self, small_hsi_cube):
        """Test L2 per-pixel normalization"""
        corrector = RadiometricCorrector(method='l2_per_pixel')
        
        result = corrector.normalize_cube(small_hsi_cube)
        
        assert result.shape == small_hsi_cube.shape
        assert result.dtype == np.float32
        
        # Check that each pixel has unit L2 norm (approximately)
        for i in range(5):
            for j in range(5):
                norm = np.linalg.norm(result[i, j, :])
                assert abs(norm - 1.0) < 0.1 or np.all(result[i, j, :] == 0)
    
    def test_minmax_normalization(self, small_hsi_cube):
        """Test min-max normalization"""
        corrector = RadiometricCorrector(method='minmax')
        
        result = corrector.normalize_cube(small_hsi_cube)
        
        assert result.shape == small_hsi_cube.shape
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_zscore_normalization(self, small_hsi_cube):
        """Test z-score normalization"""
        corrector = RadiometricCorrector(method='zscore')
        
        result = corrector.normalize_cube(small_hsi_cube)
        
        assert result.shape == small_hsi_cube.shape
        # Mean should be close to 0, std close to 1
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.2
    
    def test_percentile_normalization(self, small_hsi_cube):
        """Test percentile-based normalization"""
        corrector = RadiometricCorrector(method='percentile')
        
        result = corrector.normalize_cube(small_hsi_cube)
        
        assert result.shape == small_hsi_cube.shape
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_clip_outliers(self, small_hsi_cube):
        """Test outlier clipping"""
        corrector = RadiometricCorrector(method='minmax')
        
        result = corrector.normalize_cube(small_hsi_cube, clip_percentile=99.0)
        
        assert result.shape == small_hsi_cube.shape
    
    def test_unknown_method(self):
        """Test with unknown normalization method"""
        corrector = RadiometricCorrector(method='unknown')
        cube = np.random.rand(32, 32, 204).astype(np.float32)
        
        with pytest.raises(ValueError):
            corrector.normalize_cube(cube)


class TestHSILoader:
    """Test HSILoader class"""
    
    def test_initialization(self):
        """Test loader initialization"""
        loader = HSILoader(cache_size=10)
        assert loader.cache_size == 10
        assert len(loader.cache) == 0
    
    def test_load_from_hdf5(self, sample_hdf5):
        """Test loading from HDF5"""
        loader = HSILoader()
        
        cube, metadata = loader.load_from_hdf5(sample_hdf5, sample_idx=0)
        
        assert cube.shape == (256, 256, 204)
        assert 'sample_id' in metadata
        assert metadata['sample_id'] == 'sample_0000'
    
    def test_load_batch_from_hdf5(self, sample_hdf5):
        """Test batch loading"""
        loader = HSILoader()
        
        samples = loader.load_batch_from_hdf5(sample_hdf5, [0, 1, 2])
        
        assert len(samples) == 3
        for cube, metadata in samples:
            assert cube.shape == (256, 256, 204)
            assert 'sample_id' in metadata
    
    def test_get_hdf5_info(self, sample_hdf5):
        """Test getting HDF5 info"""
        loader = HSILoader()
        
        info = loader.get_hdf5_info(sample_hdf5)
        
        assert info['n_samples'] == 5
        assert info['cube_shape'] == (256, 256, 204)
        assert 'file_size_mb' in info
    
    def test_cache_functionality(self, sample_hdf5):
        """Test caching mechanism"""
        loader = HSILoader(cache_size=2)
        
        # Load same sample twice
        cube1, _ = loader.load_from_hdf5(sample_hdf5, 0)
        cube2, _ = loader.load_from_hdf5(sample_hdf5, 0)
        
        assert len(loader.cache) == 1
        np.testing.assert_array_equal(cube1, cube2)
    
    def test_clear_cache(self, sample_hdf5):
        """Test cache clearing"""
        loader = HSILoader()
        
        loader.load_from_hdf5(sample_hdf5, 0)
        assert len(loader.cache) > 0
        
        loader.clear_cache()
        assert len(loader.cache) == 0


class TestBatchProcessor:
    """Test BatchProcessor class"""
    
    @pytest.fixture
    def config_file(self, temp_dir, sample_config):
        """Create temporary config file"""
        import yaml
        config_path = temp_dir / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        return str(config_path)
    
    def test_initialization(self, config_file):
        """Test processor initialization"""
        processor = BatchProcessor(config_file)
        
        assert processor.batch_size == 10
        assert processor.harmonizer is not None
        assert processor.normalizer is not None
        assert processor.corrector is not None
    
    def test_process_cube(self, config_file, small_hsi_cube):
        """Test single cube processing"""
        processor = BatchProcessor(config_file)
        
        metadata = {}
        processed_cube, processed_metadata = processor.process_cube(
            cube=small_hsi_cube,
            metadata=metadata,
            source_bands=204,
            dataset_name='test'
        )
        
        assert processed_cube.shape == (256, 256, 204)
        assert processed_metadata['processed'] == True
        assert processed_metadata['target_bands'] == 204


# Integration Tests
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline"""
    
    def test_full_preprocessing_pipeline(self, small_hsi_cube):
        """Test complete preprocessing pipeline"""
        # Initialize components
        harmonizer = SpectralHarmonizer(target_bands=204)
        normalizer = SpatialNormalizer(target_size=(64, 64))
        corrector = RadiometricCorrector(method='minmax')
        
        # Process
        cube = harmonizer.harmonize_cube(small_hsi_cube)
        cube = normalizer.resize_cube(cube)
        cube = corrector.normalize_cube(cube)
        
        # Verify
        assert cube.shape == (64, 64, 204)
        assert cube.min() >= 0
        assert cube.max() <= 1
    
    def test_preprocessing_preserves_data_type(self, small_hsi_cube):
        """Test that preprocessing preserves float32 dtype"""
        harmonizer = SpectralHarmonizer(target_bands=204)
        normalizer = SpatialNormalizer(target_size=(32, 32))
        corrector = RadiometricCorrector(method='l2_per_pixel')
        
        cube = harmonizer.harmonize_cube(small_hsi_cube)
        cube = normalizer.resize_cube(cube)
        cube = corrector.normalize_cube(cube)
        
        assert cube.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
