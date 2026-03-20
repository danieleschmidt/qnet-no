"""Tests for qnetno package."""

import numpy as np
import pytest
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnetno import NVCenterChannel, QuantumNeuralOperator, DistributedQNO, ClassicalFallback


# ---- NVCenterChannel tests ----

def test_nv_channel_fidelity_perfect():
    """Channel with zero distance and zero decay should return base_fidelity."""
    ch = NVCenterChannel(base_fidelity=0.95, distance_m=0.0, decay_rate=0.1)
    assert np.isclose(ch.entanglement_fidelity(), 0.95, atol=1e-9)


def test_nv_channel_fidelity_decays_with_distance():
    """Fidelity should decay exponentially with distance."""
    base = 0.95
    rate = 0.1
    d = 5.0
    ch = NVCenterChannel(base_fidelity=base, distance_m=d, decay_rate=rate)
    expected = base * np.exp(-rate * d)
    assert np.isclose(ch.entanglement_fidelity(), expected, atol=1e-9)
    # Should be strictly less than base fidelity
    assert ch.entanglement_fidelity() < base


def test_nv_channel_above_threshold():
    """Test threshold comparison for quantum routing."""
    ch_high = NVCenterChannel(base_fidelity=0.99, distance_m=0.1, decay_rate=0.01)
    ch_low = NVCenterChannel(base_fidelity=0.99, distance_m=50.0, decay_rate=0.5)

    assert ch_high.is_above_threshold(threshold=0.85) is True
    assert ch_low.is_above_threshold(threshold=0.85) is False


def test_nv_channel_transmit_noise():
    """Transmit should return a normalized state with noise applied."""
    ch = NVCenterChannel(base_fidelity=0.95, distance_m=1.0, decay_rate=0.1)
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00> state

    noisy = ch.transmit(state)

    # Output must be a normalized state
    assert noisy.shape == state.shape
    assert np.isclose(np.linalg.norm(noisy), 1.0, atol=1e-9)
    assert noisy.dtype == complex


def test_nv_channel_transmit_preserves_norm():
    """Transmit on a multi-qubit state should always return normalized output."""
    ch = NVCenterChannel(base_fidelity=0.70, distance_m=3.0, decay_rate=0.2)
    # 3-qubit state with 8 amplitudes
    rng = np.random.default_rng(0)
    state = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    state /= np.linalg.norm(state)

    noisy = ch.transmit(state)
    assert np.isclose(np.linalg.norm(noisy), 1.0, atol=1e-9)


# ---- QuantumNeuralOperator tests ----

def test_qno_parameter_count():
    """Parameter count should equal n_layers * 2 * n_qubits."""
    qno = QuantumNeuralOperator(n_qubits=4, n_layers=2, seed=0)
    expected = 2 * 2 * 4  # n_layers * 2 (RY+RZ) * n_qubits
    assert qno.parameter_count() == expected


def test_qno_forward_shape():
    """Forward pass output shape should equal (n_qubits,)."""
    qno = QuantumNeuralOperator(n_qubits=4, n_layers=2, seed=42)
    x = np.linspace(0, 1, 16)  # 16 input features
    out = qno.forward(x)
    assert out.shape == (4,)


def test_qno_forward_deterministic():
    """Same input and same seed should give identical output."""
    qno = QuantumNeuralOperator(n_qubits=3, n_layers=2, seed=7)
    x = np.array([0.1, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6])
    out1 = qno.forward(x)
    out2 = qno.forward(x)
    np.testing.assert_array_equal(out1, out2)


def test_qno_forward_values_in_range():
    """Expectation values should lie in [-1, 1]."""
    qno = QuantumNeuralOperator(n_qubits=4, n_layers=3, seed=99)
    x = np.random.default_rng(1).standard_normal(16)
    out = qno.forward(x)
    assert np.all(out >= -1.0 - 1e-9)
    assert np.all(out <= 1.0 + 1e-9)


# ---- DistributedQNO tests ----

def test_distributed_qno_creation():
    """DistributedQNO should create the correct number of nodes and channels."""
    dqno = DistributedQNO(n_nodes=3, n_qubits_per_node=4, fidelity_threshold=0.85)
    assert len(dqno.operators) == 3
    assert len(dqno.channels) == 2  # n_nodes - 1
    assert len(dqno.fallbacks) == 3


def test_distributed_qno_forward():
    """Forward pass should return array of shape (n_qubits_per_node,)."""
    dqno = DistributedQNO(n_nodes=2, n_qubits_per_node=4, fidelity_threshold=0.85)
    x = np.linspace(0, 1, 32)
    out = dqno.forward(x)
    assert out.shape == (4,)
    assert not np.any(np.isnan(out))


def test_distributed_qno_node_fidelities():
    """node_fidelities should return a list with n_nodes-1 values."""
    dqno = DistributedQNO(n_nodes=4, n_qubits_per_node=4, fidelity_threshold=0.85)
    fids = dqno.node_fidelities()
    assert len(fids) == 3
    assert all(0.0 <= f <= 1.0 for f in fids)


# ---- ClassicalFallback tests ----

def test_classical_fallback_uses_classical_below_threshold():
    """Below threshold, output should differ from the quantum operator."""
    qno = QuantumNeuralOperator(n_qubits=4, n_layers=2, seed=42)
    fallback = ClassicalFallback(qno, threshold=0.85)

    x = np.linspace(0, 1, 16)
    out_classical = fallback.forward(x, fidelity=0.50)
    out_quantum = qno.forward(x)

    # Classical output should come from the linear layer, not quantum
    # They should NOT be identical
    assert not np.allclose(out_classical, out_quantum)
    # Classical output should be real-valued (from tanh)
    assert out_classical.dtype in (np.float64, np.float32)


def test_classical_fallback_uses_quantum_above_threshold():
    """Above threshold, output should exactly match the quantum operator."""
    qno = QuantumNeuralOperator(n_qubits=4, n_layers=2, seed=42)
    fallback = ClassicalFallback(qno, threshold=0.85)

    x = np.linspace(0, 1, 16)
    out_fallback = fallback.forward(x, fidelity=0.95)
    out_quantum = qno.forward(x)

    np.testing.assert_array_equal(out_fallback, out_quantum)


# ---- Demo integration test ----

def test_burgers_demo_runs():
    """Import demo and call main function; should run without errors."""
    import importlib.util
    demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo.py")
    spec = importlib.util.spec_from_file_location("demo", demo_path)
    demo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo_module)

    # Call main and verify it returns without exception
    result = demo_module.main()

    # Result should be a numpy array (QNO output)
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
    assert not np.any(np.isnan(result))
