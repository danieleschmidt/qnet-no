"""Classical MLP fallback when quantum channel fidelity is below threshold."""

import numpy as np
from .quantum_operator import QuantumNeuralOperator


class ClassicalFallback:
    """Automatic fallback to classical MLP when quantum fidelity < threshold.

    When the channel fidelity is sufficient (>= threshold), the quantum
    operator is used. Otherwise, a simple classical linear layer computes
    the output, avoiding the overhead and errors of a noisy quantum channel.
    """

    def __init__(self, quantum_op: QuantumNeuralOperator, threshold: float = 0.85):
        """Initialize the classical fallback wrapper.

        Args:
            quantum_op: The QuantumNeuralOperator to use when fidelity is high.
            threshold: Fidelity threshold below which the classical path is taken.
        """
        self.quantum_op = quantum_op
        self.threshold = threshold

        # Initialize a simple linear weight matrix for classical fallback
        # Output size = n_qubits (same as quantum decode output)
        rng = np.random.default_rng(seed=42)
        in_dim = quantum_op.dim
        out_dim = quantum_op.n_qubits
        # He-style initialization
        self._W = rng.standard_normal((out_dim, in_dim)).astype(np.float64) * np.sqrt(2.0 / in_dim)
        self._b = np.zeros(out_dim, dtype=np.float64)

    def forward(self, x: np.ndarray, fidelity: float) -> np.ndarray:
        """Compute forward pass, choosing quantum or classical based on fidelity.

        Args:
            x: Input array.
            fidelity: Current channel fidelity in [0, 1].

        Returns:
            Output array (same shape regardless of which path was taken).
        """
        if fidelity >= self.threshold:
            return self.quantum_op.forward(x)
        else:
            return self._classical_forward(x)

    def _classical_forward(self, x: np.ndarray) -> np.ndarray:
        """Simple linear layer: W @ x + b with tanh activation.

        Args:
            x: Input array (will be padded/truncated to match weight matrix input dim).

        Returns:
            Output array of shape (n_qubits,).
        """
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        in_dim = self._W.shape[1]

        # Pad or truncate to match weight dimensions
        x_padded = np.zeros(in_dim, dtype=np.float64)
        if len(x_flat) >= in_dim:
            x_padded[:] = x_flat[:in_dim]
        else:
            x_padded[:len(x_flat)] = x_flat

        # Normalize input
        norm = np.linalg.norm(x_padded)
        if norm > 1e-12:
            x_padded = x_padded / norm

        out = self._W @ x_padded + self._b
        return np.tanh(out)
