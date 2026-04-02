"""
Memory Profiling Utilities
Monitor memory usage during processing
"""

import psutil
import torch
import numpy as np
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """
    Profile memory usage
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage
        
        Returns:
            usage: Dictionary with memory stats
        """
        # CPU memory
        mem_info = self.process.memory_info()
        cpu_memory_mb = mem_info.rss / 1024**2
        
        # GPU memory
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
        
        # System memory
        system_mem = psutil.virtual_memory()
        system_available_mb = system_mem.available / 1024**2
        system_used_percent = system_mem.percent
        
        return {
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'system_available_mb': system_available_mb,
            'system_used_percent': system_used_percent
        }
    
    def snapshot(self, label: str = None):
        """Take memory snapshot"""
        usage = self.get_memory_usage()
        usage['label'] = label
        usage['timestamp'] = time.time()
        self.snapshots.append(usage)
        
        if label:
            logger.info(f"[{label}] CPU: {usage['cpu_memory_mb']:.1f}MB, "
                       f"GPU: {usage['gpu_memory_mb']:.1f}MB")
    
    def print_summary(self):
        """Print memory usage summary"""
        if not self.snapshots:
            logger.warning("No snapshots taken")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Memory Usage Summary")
        logger.info("="*60)
        
        for snapshot in self.snapshots:
            label = snapshot['label'] or 'Unlabeled'
            logger.info(f"{label:30s} | CPU: {snapshot['cpu_memory_mb']:8.1f}MB | "
                       f"GPU: {snapshot['gpu_memory_mb']:8.1f}MB")
        
        # Peak usage
        peak_cpu = max(s['cpu_memory_mb'] for s in self.snapshots)
        peak_gpu = max(s['gpu_memory_mb'] for s in self.snapshots)
        
        logger.info("="*60)
        logger.info(f"Peak CPU Memory: {peak_cpu:.1f}MB")
        logger.info(f"Peak GPU Memory: {peak_gpu:.1f}MB")
        logger.info("="*60 + "\n")
    
    def reset(self):
        """Reset snapshots"""
        self.snapshots = []


def profile_memory(func):
    """
    Decorator to profile memory usage of a function
    
    Usage:
        @profile_memory
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        
        profiler.snapshot(f"{func.__name__}_start")
        result = func(*args, **kwargs)
        profiler.snapshot(f"{func.__name__}_end")
        
        profiler.print_summary()
        
        return result
    
    return wrapper


def estimate_cube_memory(shape: tuple, dtype=np.float32) -> float:
    """
    Estimate memory required for HSI cube
    
    Args:
        shape: Cube shape (H, W, C)
        dtype: Data type
        
    Returns:
        memory_mb: Estimated memory in MB
    """
    n_elements = np.prod(shape)
    bytes_per_element = np.dtype(dtype).itemsize
    memory_mb = (n_elements * bytes_per_element) / 1024**2
    
    return memory_mb
