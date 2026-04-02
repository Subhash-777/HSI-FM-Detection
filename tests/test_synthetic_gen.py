"""
Test Synthetic Data Generation Module
Tests for FM insertion and synthetic data generation
"""

import pytest
import numpy as np

from src.synthetic_generation.insertion_strategies import (
    GaussianInsertion,
    EdgeBlendingInsertion,
    RealisticInsertion
)
from src.synthetic_generation.synthetic_generator import SyntheticFMGenerator
from src.synthetic_generation.augmentation import HSIAugmenter
from src.synthetic_generation.quality_checker import SyntheticQualityChecker


class TestInsertionStrategies:
    """Test FM insertion strategies"""
    
    def test_gaussian_insertion_initialization(self):
        """Test Gaussian insertion initialization"""
        inserter = GaussianInsertion(alpha_range=(0.6, 0.9))
        assert inserter.alpha_range == (0.6, 0.9)
    
    def test_gaussian_insertion(self, small_hsi_cube, sample_spectrum):
        """Test Gaussian-based FM insertion"""
        inserter = GaussianInsertion()
        
        modified_cube, mask = inserter.insert(
            cube=small_hsi_cube,
            fm_spectrum=sample_spectrum,
            center=(16, 16),
            size=3
        )
        
        assert modified_cube.shape == small_hsi_cube.shape
        assert mask.shape == (32, 32)
        assert mask.sum() > 0  # Some pixels should be marked as FM
        assert mask.max() == 1
    
    def test_edge_blending_insertion(self, small_hsi_cube, sample_spectrum):
        """Test edge-blending insertion"""
        inserter = EdgeBlendingInsertion()
        
        modified_cube, mask = inserter.insert(
            cube=small_hsi_cube,
            fm_spectrum=sample_spectrum,
            center=(16, 16),
            size=3
        )
        
        assert modified_cube.shape == small_hsi_cube.shape
        assert mask.shape == (32, 32)
        assert mask.sum() > 0
    
    def test_realistic_insertion(self, small_hsi_cube, sample_spectrum):
        """Test realistic insertion with spectral variance"""
        inserter = RealisticInsertion(spectral_noise=0.05)
        
        modified_cube, mask = inserter.insert(
            cube=small_hsi_cube,
            fm_spectrum=sample_spectrum,
            center=(16, 16),
            size=3
        )
        
        assert modified_cube.shape == small_hsi_cube.shape
        assert mask.shape == (32, 32)
        
        # Check that FM region has different spectrum
        fm_region = modified_cube[mask > 0]
        clean_region = small_hsi_cube[mask > 0]
        assert not np.allclose(fm_region, clean_region)
    
    def test_insertion_boundary_handling(self, small_hsi_cube, sample_spectrum):
        """Test insertion near image boundaries"""
        inserter = GaussianInsertion()
        
        # Insert near corner
        modified_cube, mask = inserter.insert(
            cube=small_hsi_cube,
            fm_spectrum=sample_spectrum,
            center=(2, 2),
            size=3
        )
        
        assert modified_cube.shape == small_hsi_cube.shape
        assert mask.sum() > 0


class TestSyntheticFMGenerator:
    """Test SyntheticFMGenerator class"""
    
    def test_initialization(self, sample_fm_signatures):
        """Test generator initialization"""
        generator = SyntheticFMGenerator(
            fm_signatures=sample_fm_signatures,
            insertion_strategy='realistic',
            size_range=(2, 5),
            n_objects_range=(3, 7)
        )
        
        assert len(generator.fm_classes) == 4
        assert generator.size_range == (2, 5)
        assert generator.n_objects_range == (3, 7)
    
    def test_generate_single(self, sample_fm_signatures, small_hsi_cube):
        """Test single synthetic sample generation"""
        generator = SyntheticFMGenerator(
            fm_signatures=sample_fm_signatures,
            insertion_strategy='gaussian',
            gmm_aware=False  # Disable GMM for faster testing
        )
        
        synthetic = generator.generate_single(
            clean_cube=small_hsi_cube,
            fm_class='plastic',
            n_objects=3,
            size=3
        )
        
        assert 'cube' in synthetic
        assert 'mask' in synthetic
        assert 'fm_class' in synthetic
        assert 'n_objects' in synthetic
        
        assert synthetic['cube'].shape == small_hsi_cube.shape
        assert synthetic['mask'].shape == (32, 32)
        assert synthetic['fm_class'] == 'plastic'
        assert synthetic['n_objects'] <= 3  # May be less if positions invalid
    
    def test_generate_multiple_classes(self, sample_fm_signatures, small_hsi_cube):
        """Test generation with different FM classes"""
        generator = SyntheticFMGenerator(
            fm_signatures=sample_fm_signatures,
            gmm_aware=False
        )
        
        for fm_class in ['plastic', 'textile', 'metal', 'paper']:
            synthetic = generator.generate_single(
                clean_cube=small_hsi_cube,
                fm_class=fm_class,
                n_objects=2
            )
            
            assert synthetic['fm_class'] == fm_class
    
    def test_generate_with_random_params(self, sample_fm_signatures, small_hsi_cube):
        """Test generation with random parameters"""
        generator = SyntheticFMGenerator(
            fm_signatures=sample_fm_signatures,
            gmm_aware=False
        )
        
        # Let generator choose random parameters
        synthetic = generator.generate_single(clean_cube=small_hsi_cube)
        
        assert synthetic['fm_class'] in generator.fm_classes
        assert synthetic['n_objects'] >= 0
    
    def test_non_overlapping_objects(self, sample_fm_signatures, small_hsi_cube):
        """Test that objects don't overlap significantly"""
        generator = SyntheticFMGenerator(
            fm_signatures=sample_fm_signatures,
            gmm_aware=False
        )
        
        synthetic = generator.generate_single(
            clean_cube=small_hsi_cube,
            n_objects=5,
            size=2
        )
        
        # Count connected components (should be close to n_objects)
        import cv2
        n_components, _ = cv2.connectedComponents(synthetic['mask'])
        
        # n_components includes background, so subtract 1
        assert n_components - 1 <= 5


class TestHSIAugmenter:
    """Test HSI data augmentation"""
    
    def test_initialization(self):
        """Test augmenter initialization"""
        augmenter = HSIAugmenter(
            spectral_jitter=0.05,
            band_dropout_rate=0.1,
            gaussian_noise=0.02
        )
        
        assert augmenter.spectral_jitter == 0.05
        assert augmenter.band_dropout_rate == 0.1
        assert augmenter.gaussian_noise == 0.02
    
    def test_augment(self, small_hsi_cube, small_mask):
        """Test augmentation pipeline"""
        augmenter = HSIAugmenter()
        
        aug_cube, aug_mask = augmenter.augment(small_hsi_cube, small_mask)
        
        assert aug_cube.shape == small_hsi_cube.shape
        assert aug_mask.shape == small_mask.shape
    
    def test_spectral_jitter(self, small_hsi_cube):
        """Test spectral jitter augmentation"""
        augmenter = HSIAugmenter(spectral_jitter=0.1)
        
        jittered = augmenter.add_spectral_jitter(small_hsi_cube)
        
        assert jittered.shape == small_hsi_cube.shape
        assert not np.allclose(jittered, small_hsi_cube)
        assert jittered.min() >= 0
        assert jittered.max() <= 1
    
    def test_band_dropout(self, small_hsi_cube):
        """Test band dropout augmentation"""
        augmenter = HSIAugmenter(band_dropout_rate=0.2)
        
        dropped = augmenter.band_dropout(small_hsi_cube)
        
        assert dropped.shape == small_hsi_cube.shape
        # Some bands should be zero
        zero_bands = np.sum(dropped.sum(axis=(0, 1)) == 0)
        assert zero_bands > 0
    
    def test_gaussian_noise(self, small_hsi_cube):
        """Test Gaussian noise addition"""
        augmenter = HSIAugmenter(gaussian_noise=0.05)
        
        noisy = augmenter.add_gaussian_noise(small_hsi_cube)
        
        assert noisy.shape == small_hsi_cube.shape
        assert not np.allclose(noisy, small_hsi_cube)
    
    def test_rotation(self, small_hsi_cube, small_mask):
        """Test rotation augmentation"""
        augmenter = HSIAugmenter()
        
        rotated_cube, rotated_mask = augmenter.rotate(small_hsi_cube, small_mask)
        
        assert rotated_cube.shape == small_hsi_cube.shape
        assert rotated_mask.shape == small_mask.shape
    
    def test_flip(self, small_hsi_cube, small_mask):
        """Test flip augmentation"""
        augmenter = HSIAugmenter()
        
        flipped_cube, flipped_mask = augmenter.flip(small_hsi_cube, small_mask)
        
        assert flipped_cube.shape == small_hsi_cube.shape
        assert flipped_mask.shape == small_mask.shape


class TestSyntheticQualityChecker:
    """Test quality checker for synthetic data"""
    
    def test_initialization(self):
        """Test quality checker initialization"""
        checker = SyntheticQualityChecker()
        assert checker is not None
    
    def test_check_sample(self, small_hsi_cube, small_mask):
        """Test sample quality checking"""
        checker = SyntheticQualityChecker()
        
        # Create synthetic version
        synthetic_cube = small_hsi_cube.copy()
        synthetic_cube[small_mask > 0] *= 1.5  # Modify FM region
        
        results = checker.check_sample(
            synthetic_cube=synthetic_cube,
            clean_cube=small_hsi_cube,
            mask=small_mask
        )
        
        assert 'value_range_ok' in results
        assert 'fm_stats' in results
        assert 'clean_preservation' in results
        assert 'mask_properties' in results
        assert 'quality_score' in results
        
        assert 0 <= results['quality_score'] <= 1
    
    def test_value_range_check(self, small_hsi_cube):
        """Test value range checking"""
        checker = SyntheticQualityChecker()
        
        # Valid range
        assert checker._check_value_range(small_hsi_cube) == True
        
        # Invalid range (values > 1)
        invalid_cube = small_hsi_cube * 2
        assert checker._check_value_range(invalid_cube) == False
    
    def test_fm_region_check(self, small_hsi_cube, small_mask):
        """Test FM region statistics"""
        checker = SyntheticQualityChecker()
        
        stats = checker._check_fm_region(small_hsi_cube, small_mask)
        
        assert stats['valid'] == True
        assert 'mean' in stats
        assert 'std' in stats
        assert 'n_pixels' in stats
        assert stats['n_pixels'] == small_mask.sum()


# Integration Tests
class TestSyntheticGenerationIntegration:
    """Integration tests for synthetic data generation"""
    
    def test_full_generation_pipeline(self, sample_fm_signatures, small_hsi_cube):
        """Test complete synthetic generation pipeline"""
        # Initialize generator
        generator = SyntheticFMGenerator(
            fm_signatures=sample_fm_signatures,
            insertion_strategy='realistic',
            gmm_aware=False
        )
        
        # Initialize augmenter
        augmenter = HSIAugmenter()
        
        # Initialize quality checker
        checker = SyntheticQualityChecker()
        
        # Generate synthetic sample
        synthetic = generator.generate_single(
            clean_cube=small_hsi_cube,
            fm_class='plastic'
        )
        
        # Augment
        aug_cube, aug_mask = augmenter.augment(
            synthetic['cube'],
            synthetic['mask']
        )
        
        # Check quality
        results = checker.check_sample(
            synthetic_cube=aug_cube,
            clean_cube=small_hsi_cube,
            mask=aug_mask
        )
        
        # Verify
        assert results['quality_score'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
