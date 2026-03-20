# qnet-no

**Distributed Neural Operators on Quantum Photonic Processing Units via NV-Center Entanglement**

`qnetno` simulates a distributed quantum neural operator (QNO) architecture where parameterized quantum circuits act as neural operator layers, connected across nodes via NV-center entanglement channels with realistic depolarizing noise.

## Overview

This library implements a complete simulation stack for:
- **NV-center qubit channels** with exponential fidelity decay and depolarizing noise
- **Parameterized quantum circuits** (RY/RZ gates + CNOT entanglement) as neural operator layers
- **Distributed QPU nodes** connected by NV channels with automatic classical fallback
- **1D Burgers equation demo** using the distributed QNO as a surrogate PDE solver

## Architecture

```
Input
  │
  ├──► Node 0 (QuantumNeuralOperator)
  │         │
  │    NV-Channel 0 (fidelity ~ exp(-λd))
  │         │
  ├──► Node 1 (QuantumNeuralOperator OR ClassicalFallback)
  │         │
  │    NV-Channel 1
  │         │
  └──► Node n ...
             │
        Aggregate (mean)
             │
           Output
```

When channel fidelity drops below the configured threshold, the `ClassicalFallback` (linear layer + tanh) is used in place of the quantum circuit.

## Installation

```bash
pip install -e .
```

No external quantum libraries required — uses pure NumPy state-vector simulation.

## Quick Start

```python
import numpy as np
from qnetno import DistributedQNO, NVCenterChannel, QuantumNeuralOperator, ClassicalFallback

# Create a distributed quantum neural operator
dqno = DistributedQNO(n_nodes=2, n_qubits_per_node=4, fidelity_threshold=0.85)

# Check channel fidelities
print(dqno.node_fidelities())  # [0.947..., ...]

# Process input
x = np.linspace(0, 1, 32)
out = dqno.forward(x)
print(out.shape)  # (4,) — n_qubits_per_node expectation values

# NV-center channel simulation
ch = NVCenterChannel(base_fidelity=0.95, distance_m=2.0, decay_rate=0.1)
print(ch.entanglement_fidelity())   # ~0.778
print(ch.is_above_threshold(0.85))  # False

state = np.array([1, 0, 0, 0], dtype=complex)
noisy = ch.transmit(state)
print(np.linalg.norm(noisy))  # 1.0

# Classical fallback
qno = QuantumNeuralOperator(n_qubits=4, n_layers=2, seed=42)
fb = ClassicalFallback(qno, threshold=0.85)

out_q = fb.forward(x[:16], fidelity=0.95)   # quantum path
out_c = fb.forward(x[:16], fidelity=0.50)   # classical path
```

## Demo

```bash
cd /tmp/qnet-no
python demo.py
```

Solves the 1D Burgers equation `du/dt + u*du/dx = ν d²u/dx²` with the DistributedQNO as a surrogate operator.

## Tests

```bash
pytest tests/ -v
```

## Components

### `NVCenterChannel`
Simulates a physical NV-center qubit channel:
- Fidelity: `F = F_base × exp(−λ × d)`
- Depolarizing noise: applies random Pauli errors proportional to `p = 1 − F`
- Threshold routing for quantum vs classical path selection

### `QuantumNeuralOperator`
Variational quantum circuit implemented in pure NumPy:
- Amplitude encoding of classical inputs
- Parameterized RY/RZ gates per qubit per layer
- Linear CNOT entanglement between adjacent qubits
- Pauli-Z expectation value decoding

### `DistributedQNO`
Multi-node quantum processing architecture:
- Splits input across `n_nodes` QPU nodes
- Each node-to-node link simulated as NV-center channel
- Automatic classical fallback per node based on fidelity
- Output aggregation via mean across nodes

### `ClassicalFallback`
Transparent quantum/classical switch:
- Routes to `QuantumNeuralOperator` when fidelity ≥ threshold
- Falls back to linear layer (numpy matmul + tanh) when below threshold
- Same interface regardless of path taken

## Physical Motivation

NV (nitrogen-vacancy) centers in diamond are leading candidates for quantum network nodes due to their long coherence times and ability to entangle via photonic interfaces. This library models the key limitation of such systems: fidelity loss over distance, and the resulting need for classical fallback strategies in near-term quantum networks.

## License

MIT
