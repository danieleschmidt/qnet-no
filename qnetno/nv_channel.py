"""NV-center qubit channel simulation with depolarizing noise."""

import numpy as np


class NVCenterChannel:
    """Simulates NV-center qubit channel with depolarizing noise.

    The channel applies a depolarizing noise model where fidelity
    decays exponentially with distance:
        fidelity = base_fidelity * exp(-decay_rate * distance_m)

    Depolarizing channel: rho -> (1-p)*rho + p/4 * I
    where p = 1 - fidelity (for single qubit; generalized for n qubits).
    """

    def __init__(self, base_fidelity: float = 0.95, distance_m: float = 1.0, decay_rate: float = 0.1):
        """Initialize the NV-center channel.

        Args:
            base_fidelity: Maximum fidelity at zero distance (0 < F <= 1).
            distance_m: Physical distance in meters.
            decay_rate: Exponential decay rate (1/m).
        """
        if not (0 < base_fidelity <= 1.0):
            raise ValueError("base_fidelity must be in (0, 1]")
        if distance_m < 0:
            raise ValueError("distance_m must be non-negative")
        if decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")

        self.base_fidelity = base_fidelity
        self.distance_m = distance_m
        self.decay_rate = decay_rate
        self._fidelity = base_fidelity * np.exp(-decay_rate * distance_m)

    def entanglement_fidelity(self) -> float:
        """Return the entanglement fidelity F in [0, 1]."""
        return float(self._fidelity)

    def is_above_threshold(self, threshold: float = 0.85) -> bool:
        """Return True if channel fidelity exceeds the threshold."""
        return bool(self._fidelity >= threshold)

    def transmit(self, state_vector: np.ndarray) -> np.ndarray:
        """Apply depolarizing noise to a state vector and return the noisy state.

        Models the depolarizing channel acting on the density matrix:
            rho -> (1-p)*rho + p/d * I
        where d = len(state_vector) and p = 1 - fidelity.

        For the state-vector representation, we apply random Pauli errors
        proportional to the noise parameter p.

        Args:
            state_vector: Complex numpy array representing a quantum state.
                          Does not need to be normalized.

        Returns:
            Noisy state vector (normalized).
        """
        state = np.array(state_vector, dtype=complex)
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm

        p = 1.0 - self._fidelity  # depolarizing error probability
        n = len(state)
        n_qubits = int(np.round(np.log2(n)))

        rng = np.random.default_rng(seed=None)

        # Apply random single-qubit Pauli errors with probability p per qubit
        noisy_state = state.copy()
        for q in range(n_qubits):
            if rng.random() < p:
                # Pick random Pauli: X, Y, or Z with equal probability
                pauli_choice = rng.integers(0, 3)
                noisy_state = _apply_single_qubit_gate(noisy_state, pauli_choice, q, n_qubits)

        # Re-normalize
        norm_out = np.linalg.norm(noisy_state)
        if norm_out > 0:
            noisy_state = noisy_state / norm_out

        return noisy_state


def _apply_single_qubit_gate(state: np.ndarray, pauli: int, qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a Pauli gate (0=X, 1=Y, 2=Z) to a specific qubit in the state vector."""
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [X, Y, Z]
    gate = paulis[pauli]

    # Build full operator via tensor product
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate @ state
