"""
Synthetic Data Generation Module
Generate synthetic FM-inserted HSI cubes
"""

from .synthetic_generator import SyntheticFMGenerator
from .insertion_strategies import InsertionStrategy, GaussianInsertion, EdgeBlendingInsertion
from .augmentation import HSIAugmenter
from .quality_checker import SyntheticQualityChecker

__all__ = [
    'SyntheticFMGenerator',
    'InsertionStrategy',
    'GaussianInsertion',
    'EdgeBlendingInsertion',
    'HSIAugmenter',
    'SyntheticQualityChecker'
]
