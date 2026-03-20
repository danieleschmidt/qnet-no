"""Parameterized quantum circuit as a neural operator layer."""

import numpy as np
from typing import Optional


class QuantumNeuralOperator:
    """Parameterized quantum circuit as neural operator layer.

    Implements a variational quantum circuit (VQC) using pure numpy
    state-vector simulation. The circuit consists of alternating layers
    of single-qubit rotations (RY, RZ) and entangling CNOT gates.

    Gate definitions:
        RY(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        RZ(t) = [[exp(-it/2), 0], [0, exp(it/2)]]
        CNOT  = standard 4x4 matrix
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, seed: int = 42):
        """Initialize the quantum neural operator.

        Args:
            n_qubits: Number of qubits (state space = 2**n_qubits).
            n_layers: Number of variational layers.
            seed: Random seed for parameter initialization.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        rng = np.random.default_rng(seed)
        # Each layer has RY + RZ per qubit: 2 * n_qubits angles per layer
        self.params = rng.uniform(0, 2 * np.pi, size=(n_layers, 2, n_qubits))
        # params[layer, 0, qubit] = RY angle
        # params[layer, 1, qubit] = RZ angle

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return int(self.params.size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Process input through the quantum circuit.

        Pipeline: encode -> circuit -> decode

        Args:
            x: 1D numpy array of real values (input features).

        Returns:
            1D numpy array of expectation values (output features).
        """
        state = self._encode(x)
        state = self._circuit(state)
        return self._decode(state)

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Amplitude encoding of input into quantum state.

        Pads or truncates x to length 2**n_qubits and normalizes.

        Args:
            x: Input array.

        Returns:
            Normalized complex state vector of shape (2**n_qubits,).
        """
        x_flat = np.asarray(x, dtype=complex).flatten()
        state = np.zeros(self.dim, dtype=complex)

        if len(x_flat) >= self.dim:
            state[:] = x_flat[:self.dim]
        else:
            state[:len(x_flat)] = x_flat

        norm = np.linalg.norm(state)
        if norm < 1e-12:
            state[0] = 1.0
        else:
            state = state / norm

        return state

    def _circuit(self, state: np.ndarray) -> np.ndarray:
        """Apply parameterized quantum circuit gates.

        Each layer consists of:
          1. RY rotations on all qubits
          2. RZ rotations on all qubits
          3. CNOT entangling gates on adjacent pairs

        Args:
            state: Complex state vector of shape (2**n_qubits,).

        Returns:
            Transformed state vector.
        """
        for layer in range(self.n_layers):
            # RY gates
            for q in range(self.n_qubits):
                theta = self.params[layer, 0, q]
                gate = _ry(theta)
                state = _apply_single_qubit(state, gate, q, self.n_qubits)

            # RZ gates
            for q in range(self.n_qubits):
                phi = self.params[layer, 1, q]
                gate = _rz(phi)
                state = _apply_single_qubit(state, gate, q, self.n_qubits)

            # CNOT entangling layer (linear chain)
            for q in range(self.n_qubits - 1):
                state = _apply_cnot(state, q, q + 1, self.n_qubits)

        return state

    def _decode(self, state: np.ndarray) -> np.ndarray:
        """Measure Pauli-Z expectation values for each qubit.

        <Z_q> = sum_{x} |psi_x|^2 * (-1)^{bit_q(x)}

        Args:
            state: Complex state vector.

        Returns:
            Array of n_qubits expectation values in [-1, 1].
        """
        probs = np.abs(state) ** 2
        expectations = np.zeros(self.n_qubits)

        for q in range(self.n_qubits):
            for idx in range(self.dim):
                # Check the q-th bit of idx (MSB ordering)
                bit = (idx >> (self.n_qubits - 1 - q)) & 1
                sign = 1 - 2 * bit  # bit=0 -> +1, bit=1 -> -1
                expectations[q] += sign * probs[idx]

        return expectations


# ---- Gate helpers ----

def _ry(theta: float) -> np.ndarray:
    """2x2 RY rotation matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(phi: float) -> np.ndarray:
    """2x2 RZ rotation matrix."""
    return np.array([
        [np.exp(-1j * phi / 2), 0],
        [0, np.exp(1j * phi / 2)]
    ], dtype=complex)


def _apply_single_qubit(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 gate to a specific qubit using tensor product construction."""
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full @ state


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """Apply CNOT gate with given control and target qubits."""
    dim = 2 ** n_qubits
    cnot_full = np.eye(dim, dtype=complex)

    for idx in range(dim):
        ctrl_bit = (idx >> (n_qubits - 1 - control)) & 1
        if ctrl_bit == 1:
            # Flip the target bit
            tgt_mask = 1 << (n_qubits - 1 - target)
            flipped = idx ^ tgt_mask
            # Swap rows
            cnot_full[idx, idx] = 0
            cnot_full[flipped, idx] = 1
            cnot_full[idx, flipped] = 1
            cnot_full[flipped, flipped] = 0

    return cnot_full @ state
