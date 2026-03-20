"""qnetno - Distributed Neural Operators on Quantum Photonic Processing Units.

Simulates distributed quantum neural operators using NV-center entanglement
channels and parameterized quantum circuits implemented in pure numpy.
"""

from .nv_channel import NVCenterChannel
from .quantum_operator import QuantumNeuralOperator
from .distributed_qno import DistributedQNO
from .classical_fallback import ClassicalFallback

__all__ = [
    "NVCenterChannel",
    "QuantumNeuralOperator",
    "DistributedQNO",
    "ClassicalFallback",
]

__version__ = "0.1.0"
