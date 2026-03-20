"""Distributed Quantum Neural Operator across simulated QPU nodes."""

import numpy as np
from typing import List
from .nv_channel import NVCenterChannel
from .quantum_operator import QuantumNeuralOperator
from .classical_fallback import ClassicalFallback


class DistributedQNO:
    """Split quantum neural operator across simulated QPU nodes connected by NV channels.

    Each node hosts its own QuantumNeuralOperator. Adjacent nodes are connected
    by NV-center channels. When channel fidelity drops below the threshold, the
    ClassicalFallback is used for that node's computation.
    """

    def __init__(
        self,
        n_nodes: int = 2,
        n_qubits_per_node: int = 4,
        fidelity_threshold: float = 0.85,
    ):
        """Initialize the distributed QNO.

        Args:
            n_nodes: Number of QPU nodes.
            n_qubits_per_node: Qubits per node.
            fidelity_threshold: Minimum fidelity to use quantum path.
        """
        if n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")

        self.n_nodes = n_nodes
        self.n_qubits_per_node = n_qubits_per_node
        self.fidelity_threshold = fidelity_threshold

        # One quantum operator per node
        self.operators: List[QuantumNeuralOperator] = [
            QuantumNeuralOperator(n_qubits=n_qubits_per_node, n_layers=2, seed=i * 7 + 42)
            for i in range(n_nodes)
        ]

        # NV channels connecting adjacent nodes (n_nodes - 1 channels)
        # Each channel has slightly different distance to simulate realistic variation
        self.channels: List[NVCenterChannel] = [
            NVCenterChannel(
                base_fidelity=0.97,
                distance_m=0.5 + i * 0.3,  # increasing distance per hop
                decay_rate=0.1,
            )
            for i in range(max(n_nodes - 1, 0))
        ]

        # Classical fallbacks for each node
        self.fallbacks: List[ClassicalFallback] = [
            ClassicalFallback(op, threshold=fidelity_threshold)
            for op in self.operators
        ]

    def node_fidelities(self) -> List[float]:
        """Return list of fidelity for each node-to-node channel.

        Returns:
            List of length (n_nodes - 1) with channel fidelities.
        """
        return [ch.entanglement_fidelity() for ch in self.channels]

    def _route(self, x: np.ndarray) -> List[bool]:
        """Decide which nodes use quantum vs classical path.

        Node 0 always uses quantum (no incoming channel).
        Node i uses quantum if the channel from node i-1 to i is above threshold.

        Args:
            x: Input array (unused for routing decision in this model).

        Returns:
            List of booleans (True = use quantum) for each node.
        """
        use_quantum = [True]  # Node 0 always quantum

        for ch in self.channels:
            use_quantum.append(ch.is_above_threshold(self.fidelity_threshold))

        return use_quantum

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Process input through distributed QPU nodes.

        Splits input into n_nodes chunks, processes each chunk on its
        respective node (quantum or classical fallback), then aggregates
        (mean) the outputs.

        Args:
            x: Input array (will be split across nodes).

        Returns:
            Aggregated output array of shape (n_qubits_per_node,).
        """
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        use_quantum = self._route(x_flat)

        # Split input across nodes
        chunks = np.array_split(x_flat, self.n_nodes)

        outputs = []
        fidelities = self.node_fidelities() if self.channels else []

        for i, (chunk, fallback) in enumerate(zip(chunks, self.fallbacks)):
            # Get fidelity for this node's incoming channel
            if i == 0:
                fid = 1.0  # First node: no channel, assume perfect
            else:
                fid = fidelities[i - 1]

            out = fallback.forward(chunk, fidelity=fid)
            outputs.append(out)

        # Aggregate: mean across nodes
        return np.mean(outputs, axis=0)
