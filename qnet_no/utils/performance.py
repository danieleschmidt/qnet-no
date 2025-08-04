"""Performance optimization utilities for QNet-NO."""

import time
import functools
from typing import Dict, Any, Optional, Callable, Tuple, List
import jax
import jax.numpy as jnp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import hashlib
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""
    
    def __init__(self, max_size_mb: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.pool: Dict[Tuple[tuple, jnp.dtype], List[jnp.ndarray]] = {}
        self.current_size = 0
        self.lock = threading.Lock()
    
    def get_tensor(self, shape: tuple, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
        """Get a tensor from the pool or allocate new one."""
        key = (shape, dtype)
        
        with self.lock:
            if key in self.pool and self.pool[key]:
                tensor = self.pool[key].pop()
                logger.debug(f"Reused tensor from pool: shape={shape}, dtype={dtype}")
                return tensor
        
        # Allocate new tensor
        tensor = jnp.zeros(shape, dtype=dtype)
        logger.debug(f"Allocated new tensor: shape={shape}, dtype={dtype}")
        return tensor
    
    def return_tensor(self, tensor: jnp.ndarray):
        """Return a tensor to the pool for reuse."""
        if tensor.size * tensor.dtype.itemsize > self.max_size_bytes // 10:
            # Don't pool very large tensors
            return
        
        key = (tensor.shape, tensor.dtype)
        
        with self.lock:
            if key not in self.pool:
                self.pool[key] = []
            
            # Only keep a few tensors of each type
            if len(self.pool[key]) < 5:
                # Zero out the tensor for security
                tensor = jnp.zeros_like(tensor)
                self.pool[key].append(tensor)
                self.current_size += tensor.size * tensor.dtype.itemsize
                
                # Clean up if pool is too large
                if self.current_size > self.max_size_bytes:
                    self._cleanup_pool()
    
    def _cleanup_pool(self):
        """Clean up the pool to free memory."""
        # Remove half the tensors from each type
        for key in list(self.pool.keys()):
            if self.pool[key]:
                removed_count = len(self.pool[key]) // 2
                for _ in range(removed_count):
                    if self.pool[key]:
                        tensor = self.pool[key].pop()
                        self.current_size -= tensor.size * tensor.dtype.itemsize
        
        logger.info(f"Memory pool cleaned up, current size: {self.current_size / 1024**2:.1f}MB")


class ComputationCache:
    """Cache for expensive quantum computations."""
    
    def __init__(self, max_size: int = 1000, cache_dir: Optional[Path] = None):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_key(self, *args, **kwargs) -> str:
        """Compute cache key from arguments."""
        # Create hashable representation
        key_data = {
            'args': [self._make_hashable(arg) for arg in args],
            'kwargs': {k: self._make_hashable(v) for k, v in kwargs.items()}
        }
        
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_str).hexdigest()
    
    def _make_hashable(self, obj):
        """Convert object to hashable representation."""
        if isinstance(obj, jnp.ndarray):
            return ('array', obj.shape, obj.dtype, hash(obj.tobytes()))
        elif isinstance(obj, np.ndarray):
            return ('numpy_array', obj.shape, obj.dtype, hash(obj.tobytes()))
        elif isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, (list, tuple)):
            return tuple(self._make_hashable(item) for item in obj)
        else:
            return obj
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                # Update access order (LRU)
                self.access_order.remove(key)
                self.access_order.append(key)
                logger.debug(f"Cache hit (memory): {key[:16]}...")
                return self.memory_cache[key]
        
        # Check disk cache if available
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Add to memory cache
                    with self.lock:
                        self._add_to_memory_cache(key, result)
                    
                    logger.debug(f"Cache hit (disk): {key[:16]}...")
                    return result
                    
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")
        
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            self._add_to_memory_cache(key, value)
        
        # Save to disk cache if available
        if self.cache_dir:
            try:
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"Cached to disk: {key[:16]}...")
            except Exception as e:
                logger.warning(f"Failed to save to disk cache: {e}")
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """Add item to memory cache with LRU eviction."""
        if key in self.memory_cache:
            self.access_order.remove(key)
        
        self.memory_cache[key] = value
        self.access_order.append(key)
        
        # Evict oldest items if cache is full
        while len(self.memory_cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.memory_cache[oldest_key]
        
        logger.debug(f"Cached in memory: {key[:16]}...")
    
    def clear(self):
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            self.access_order.clear()
        
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")


# Global instances
memory_pool = MemoryPool()
computation_cache = ComputationCache(cache_dir=Path.home() / ".qnet_no" / "cache")


def cached_computation(cache_key_func: Optional[Callable] = None, 
                      expire_after: Optional[float] = None):
    """
    Decorator for caching expensive quantum computations.
    
    Args:
        cache_key_func: Function to compute custom cache key
        expire_after: Cache expiration time in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = computation_cache._compute_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = computation_cache.get(cache_key)
            if cached_result is not None:
                if expire_after is None:
                    return cached_result
                
                # Check expiration
                cached_time, result = cached_result
                if time.time() - cached_time < expire_after:
                    return result
            
            # Compute result
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Cache result
            if expire_after is not None:
                cached_data = (time.time(), result)
            else:
                cached_data = result
            
            computation_cache.put(cache_key, cached_data)
            
            logger.debug(f"Computed and cached {func.__name__} in {computation_time:.3f}s")
            return result
        
        return wrapper
    return decorator


def parallel_map(func: Callable, items: List[Any], max_workers: Optional[int] = None,
                use_processes: bool = False) -> List[Any]:
    """
    Apply function to items in parallel.
    
    Args:
        func: Function to apply
        items: List of items to process
        max_workers: Maximum number of workers
        use_processes: Use processes instead of threads
    
    Returns:
        List of results
    """
    if len(items) <= 1:
        return [func(item) for item in items]
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results


def batch_process(func: Callable, data: jnp.ndarray, batch_size: int = 32,
                 overlap: int = 0) -> jnp.ndarray:
    """
    Process data in batches to manage memory usage.
    
    Args:
        func: Function to apply to each batch
        data: Input data tensor
        batch_size: Size of each batch
        overlap: Overlap between batches
        
    Returns:
        Concatenated results
    """
    if data.shape[0] <= batch_size:
        return func(data)
    
    results = []
    n_samples = data.shape[0]
    
    for start_idx in range(0, n_samples, batch_size - overlap):
        end_idx = min(start_idx + batch_size, n_samples)
        
        batch = data[start_idx:end_idx]
        batch_result = func(batch)
        
        # Handle overlap by trimming results
        if overlap > 0 and start_idx > 0:
            batch_result = batch_result[overlap:]
        
        results.append(batch_result)
    
    return jnp.concatenate(results, axis=0)


class PerformanceProfiler:
    """Profiler for quantum operations performance analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self.current_operation: Optional[str] = None
        self.start_time: Optional[float] = None
    
    def start_operation(self, operation_name: str, **metadata):
        """Start profiling an operation."""
        self.current_operation = operation_name
        self.start_time = time.time()
        
        if operation_name not in self.profiles:
            self.profiles[operation_name] = []
        
        # Store metadata
        self.current_metadata = metadata
    
    def end_operation(self, **additional_metadata):
        """End profiling current operation."""
        if self.current_operation is None or self.start_time is None:
            logger.warning("No operation currently being profiled")
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        profile_entry = {
            'start_time': self.start_time,
            'end_time': end_time,
            'duration': duration,
            **self.current_metadata,
            **additional_metadata
        }
        
        self.profiles[self.current_operation].append(profile_entry)
        
        logger.debug(f"Operation {self.current_operation} completed in {duration:.3f}s")
        
        self.current_operation = None
        self.start_time = None
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation_name not in self.profiles:
            return {}
        
        durations = [entry['duration'] for entry in self.profiles[operation_name]]
        
        return {
            'count': len(durations),
            'total_time': sum(durations),
            'avg_time': np.mean(durations),
            'min_time': np.min(durations),
            'max_time': np.max(durations),
            'std_time': np.std(durations)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_statistics(op) for op in self.profiles.keys()}


def profile_operation(operation_name: str, **metadata):
    """Decorator to profile operation performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create profiler
            if not hasattr(wrapper, '_profiler'):
                wrapper._profiler = PerformanceProfiler()
            
            profiler = wrapper._profiler
            
            # Start profiling
            profiler.start_operation(operation_name, **metadata)
            
            try:
                result = func(*args, **kwargs)
                
                # Add result metadata if applicable
                result_metadata = {}
                if hasattr(result, 'shape'):
                    result_metadata['output_shape'] = result.shape
                
                profiler.end_operation(**result_metadata)
                return result
                
            except Exception as e:
                profiler.end_operation(error=str(e))
                raise
        
        # Add method to get statistics
        wrapper.get_profile_stats = lambda: wrapper._profiler.get_all_statistics()
        
        return wrapper
    return decorator


def optimize_memory_layout(tensor: jnp.ndarray, target_layout: str = "row_major") -> jnp.ndarray:
    """
    Optimize tensor memory layout for better cache performance.
    
    Args:
        tensor: Input tensor
        target_layout: Target memory layout
        
    Returns:
        Tensor with optimized layout
    """
    if target_layout == "row_major":
        # Ensure C-contiguous layout
        if not tensor.flags.c_contiguous:
            tensor = jnp.ascontiguousarray(tensor)
    elif target_layout == "col_major":
        # Ensure Fortran-contiguous layout
        if not tensor.flags.f_contiguous:
            tensor = jnp.asfortranarray(tensor)
    
    return tensor


class AdaptiveBatchSize:
    """Dynamically adjust batch size based on available memory and performance."""
    
    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 1, 
                 max_batch_size: int = 512):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.performance_history = []
        self.memory_limit_hit = False
    
    def get_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.current_batch_size
    
    def record_performance(self, batch_size: int, execution_time: float, 
                          memory_used: float, success: bool):
        """Record performance metrics for batch size adaptation."""
        self.performance_history.append({
            'batch_size': batch_size,
            'execution_time': execution_time,
            'memory_used': memory_used,
            'success': success,
            'throughput': batch_size / execution_time if execution_time > 0 else 0
        })
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        
        # Adapt batch size
        self._adapt_batch_size(success)
    
    def _adapt_batch_size(self, last_success: bool):
        """Adapt batch size based on performance history."""
        if not last_success:
            # Reduce batch size if last operation failed
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.7)
            )
            self.memory_limit_hit = True
            logger.info(f"Reduced batch size to {self.current_batch_size} due to failure")
            return
        
        # If we have enough history, analyze trends
        if len(self.performance_history) < 5:
            return
        
        recent_entries = self.performance_history[-5:]
        avg_throughput = np.mean([entry['throughput'] for entry in recent_entries])
        
        # Compare with earlier performance
        if len(self.performance_history) >= 10:
            earlier_entries = self.performance_history[-10:-5]
            earlier_throughput = np.mean([entry['throughput'] for entry in earlier_entries])
            
            if avg_throughput > earlier_throughput * 1.1:
                # Performance is improving, try larger batch size
                if not self.memory_limit_hit and self.current_batch_size < self.max_batch_size:
                    self.current_batch_size = min(
                        self.max_batch_size,
                        int(self.current_batch_size * 1.2)
                    )
                    logger.debug(f"Increased batch size to {self.current_batch_size}")
            
            elif avg_throughput < earlier_throughput * 0.9:
                # Performance is degrading, try smaller batch size
                self.current_batch_size = max(
                    self.min_batch_size,
                    int(self.current_batch_size * 0.9)
                )
                logger.debug(f"Decreased batch size to {self.current_batch_size}")


def warmup_jax_compilation(shapes: List[tuple], dtypes: List[jnp.dtype] = None):
    """
    Warm up JAX compilation for common tensor shapes.
    
    Args:
        shapes: List of tensor shapes to pre-compile
        dtypes: List of data types (defaults to float32)
    """
    if dtypes is None:
        dtypes = [jnp.float32] * len(shapes)
    
    logger.info(f"Warming up JAX compilation for {len(shapes)} shapes")
    
    # Pre-compile common operations
    @jax.jit
    def warmup_ops(x):
        # Common operations in quantum neural operators
        y = jnp.fft.fft2(x)
        z = jnp.real(y)
        w = jnp.dot(z.reshape(z.shape[0], -1), z.reshape(z.shape[0], -1).T)
        return w
    
    for shape, dtype in zip(shapes, dtypes):
        try:
            dummy_tensor = jnp.ones(shape, dtype=dtype)
            _ = warmup_ops(dummy_tensor)
            logger.debug(f"Warmed up shape {shape} with dtype {dtype}")
        except Exception as e:
            logger.warning(f"Failed to warm up shape {shape}: {e}")
    
    logger.info("JAX warmup completed")


# Global adaptive batch size manager
adaptive_batch_manager = AdaptiveBatchSize()

# Global performance profiler
global_profiler = PerformanceProfiler()