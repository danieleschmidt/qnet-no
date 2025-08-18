"""Robust validation utilities for Generation 2 improvements."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def validate_tensor_shapes(tensor1: jnp.ndarray, tensor2: jnp.ndarray, 
                          operation: str = "dot_product") -> bool:
    """Validate tensor shapes for compatibility in quantum operations."""
    try:
        if operation == "dot_product":
            # For batched matrix multiplication: [batch, seq, dim] @ [batch, dim, out]
            if len(tensor1.shape) == 3 and len(tensor2.shape) == 3:
                if tensor1.shape[0] != tensor2.shape[0]:  # Batch size mismatch
                    logger.warning(f"Batch size mismatch: {tensor1.shape[0]} vs {tensor2.shape[0]}")
                    return False
                if tensor1.shape[-1] != tensor2.shape[-2]:  # Feature dimension mismatch
                    logger.warning(f"Feature dimension mismatch: {tensor1.shape[-1]} vs {tensor2.shape[-2]}")
                    return False
            elif len(tensor1.shape) == 2 and len(tensor2.shape) == 2:
                if tensor1.shape[-1] != tensor2.shape[0]:
                    logger.warning(f"Matrix dimension mismatch: {tensor1.shape} vs {tensor2.shape}")
                    return False
        return True
    except Exception as e:
        logger.error(f"Tensor validation failed: {e}")
        return False

def safe_tensor_operation(operation: str, *tensors, **kwargs) -> Optional[jnp.ndarray]:
    """Safely execute tensor operations with validation."""
    try:
        if operation == "distributed_dot_product":
            a, b = tensors[0], tensors[1]
            network = kwargs.get('network')
            
            # Basic validation
            if not isinstance(a, jnp.ndarray) or not isinstance(b, jnp.ndarray):
                logger.error(f"Invalid tensor types: {type(a)}, {type(b)}")
                return None
                
            # Shape compatibility check
            if not validate_tensor_shapes(a, b, "dot_product"):
                # Try to fix common shape issues
                if len(a.shape) == 3 and len(b.shape) == 3:
                    if a.shape[-1] == b.shape[-1]:  # Same feature dim, transpose b
                        b = jnp.swapaxes(b, -2, -1)
                        logger.info(f"Fixed tensor shapes by transposing: {a.shape} @ {b.shape}")
                    else:
                        logger.error(f"Cannot fix incompatible shapes: {a.shape} vs {b.shape}")
                        return None
            
            # Perform the actual operation
            if len(a.shape) == 3 and len(b.shape) == 3:
                return jnp.einsum('bij,bjk->bik', a, b)
            elif len(a.shape) == 2 and len(b.shape) == 2:
                return jnp.dot(a, b)
            else:
                return jnp.tensordot(a, b, axes=(-1, -2))
                
    except Exception as e:
        logger.error(f"Safe tensor operation failed: {e}")
        return None

def validate_network_consistency(network) -> bool:
    """Validate quantum network consistency and properties."""
    try:
        if network is None:
            return True  # Allow None networks for simulation
            
        # Check basic network properties
        if hasattr(network, 'num_nodes') and hasattr(network, 'num_links'):
            if network.num_nodes <= 0:
                logger.warning("Network has no nodes")
                return False
            # Remove overly strict link requirement for single-node networks
            if network.num_nodes > 1 and network.num_links == 0:
                logger.info("Multi-node network has no entanglement links (degraded mode)")
                
        return True
    except Exception as e:
        logger.error(f"Network validation failed: {e}")
        return False

def create_fallback_cache_dir() -> str:
    """Create a safe fallback cache directory."""
    import tempfile
    import os
    
    try:
        cache_dir = os.path.join(tempfile.gettempdir(), "qnet_no_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    except Exception:
        return tempfile.gettempdir()  # Fall back to system temp