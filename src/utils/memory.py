"""
Memory management utilities.
Prevents memory leaks and optimizes resource usage.
"""
import gc
import os
import psutil
from functools import wraps
from typing import Optional, Callable, Any
from contextlib import contextmanager

import torch
import numpy as np
from loguru import logger


class MemoryManager:
    """
    Memory management class for efficient resource handling.
    Monitors memory usage and triggers cleanup when needed.
    """
    
    def __init__(self, max_memory_mb: int = 4096, warning_threshold: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum allowed memory in MB
            warning_threshold: Percentage of max memory to trigger warning
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = warning_threshold
        self._process = psutil.Process(os.getpid())
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._process.memory_info().rss / (1024 * 1024)
    
    def get_gpu_memory_usage_mb(self) -> Optional[float]:
        """Get GPU memory usage in MB (if available)."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return None
    
    def check_memory(self) -> bool:
        """
        Check if memory usage is within limits.
        Returns False if memory exceeds threshold.
        """
        current_mb = self.get_memory_usage_mb()
        threshold_mb = self.max_memory_mb * self.warning_threshold
        
        if current_mb > threshold_mb:
            logger.warning(f"Memory usage high: {current_mb:.1f}MB / {self.max_memory_mb}MB")
            return False
        return True
    
    def cleanup(self, force: bool = False) -> None:
        """
        Perform aggressive memory cleanup.
        
        Args:
            force: Force garbage collection even if memory is within limits
        """
        if force or not self.check_memory():
            # Clear Python garbage (multiple passes for better cleanup)
            for _ in range(2):
                collected = gc.collect()
                if collected == 0:
                    break
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force GPU garbage collection
                torch.cuda.ipc_collect()
            
            logger.debug(f"Memory cleanup completed. Current: {self.get_memory_usage_mb():.1f}MB")
    
    def cleanup_intermediate_data(self, *objects) -> None:
        """
        Explicitly delete intermediate objects to free memory immediately.
        
        Args:
            *objects: Objects to delete
        """
        for obj in objects:
            del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def log_memory_status(self) -> None:
        """Log current memory status."""
        cpu_mb = self.get_memory_usage_mb()
        gpu_mb = self.get_gpu_memory_usage_mb()
        
        status = f"CPU Memory: {cpu_mb:.1f}MB"
        if gpu_mb is not None:
            status += f" | GPU Memory: {gpu_mb:.1f}MB"
        
        logger.info(status)


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(max_memory_mb: int = 4096) -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(max_memory_mb=max_memory_mb)
    return _memory_manager


def memory_efficient(cleanup_after: bool = True):
    """
    Decorator for memory-efficient function execution.
    Cleans up memory after function execution.
    
    Args:
        cleanup_after: Whether to cleanup memory after function execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            manager = get_memory_manager()
            
            # Check memory before execution
            manager.check_memory()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if cleanup_after:
                    manager.cleanup()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            manager = get_memory_manager()
            
            # Check memory before execution
            manager.check_memory()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                if cleanup_after:
                    manager.cleanup()
        
        # Return appropriate wrapper based on function type
        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is a coroutine function."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


@contextmanager
def managed_array(shape: tuple, dtype=np.float32):
    """
    Context manager for numpy arrays with automatic cleanup.
    
    Usage:
        with managed_array((1000, 1000), dtype=np.float32) as arr:
            # Use arr
        # arr is automatically cleaned up
    """
    arr = np.zeros(shape, dtype=dtype)
    try:
        yield arr
    finally:
        del arr
        gc.collect()


@contextmanager
def managed_tensor(shape: tuple, dtype=torch.float32, device: str = "cpu"):
    """
    Context manager for PyTorch tensors with automatic cleanup.
    
    Usage:
        with managed_tensor((1000, 1000), device="cuda") as tensor:
            # Use tensor
        # tensor is automatically cleaned up
    """
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    try:
        yield tensor
    finally:
        del tensor
        if device == "cuda" or device.startswith("cuda:"):
            torch.cuda.empty_cache()
        gc.collect()

