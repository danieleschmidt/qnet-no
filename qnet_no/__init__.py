"""
QNet-NO: Quantum-Network Neural Operator Library

A library for distributed quantum neural operators running on photonic QPUs
connected via entanglement channels.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .operators import (
    QuantumFourierNeuralOperator,
    QuantumDeepONet,
    HybridNeuralOperator,
)
from .networks import PhotonicNetwork, EntanglementScheduler
from .backends import PhotonicBackend, NVCenterBackend, SimulatorBackend
from . import datasets

__all__ = [
    "QuantumFourierNeuralOperator", 
    "QuantumDeepONet",
    "HybridNeuralOperator",
    "PhotonicNetwork",
    "EntanglementScheduler", 
    "PhotonicBackend",
    "NVCenterBackend",
    "SimulatorBackend",
    "datasets",
]