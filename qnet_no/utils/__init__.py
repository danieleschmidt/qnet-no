"""Utility functions for quantum neural operators."""

from .quantum_fourier import quantum_fourier_modes, inverse_quantum_fourier_transform
from .quantum_encoding import quantum_feature_map, amplitude_encoding
from .tensor_ops import tensor_product_einsum, distributed_dot_product
from .classical_layers import ResidualBlock, AttentionLayer

__all__ = [
    "quantum_fourier_modes",
    "inverse_quantum_fourier_transform", 
    "quantum_feature_map",
    "amplitude_encoding",
    "tensor_product_einsum",
    "distributed_dot_product",
    "ResidualBlock",
    "AttentionLayer",
]