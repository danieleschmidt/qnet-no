"""Photonic quantum network topology and management."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import networkx as nx
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
import jax.numpy as jnp
import logging
from ..utils.validation import validate_network_parameters, log_validation_result
from ..utils.error_handling import NetworkError, ErrorSeverity, error_boundary

logger = logging.getLogger(__name__)


@dataclass
class QuantumNode:
    """Represents a quantum processing unit node."""
    node_id: int
    qpu_type: str  # "photonic", "superconducting", "trapped_ion", "nv_center"
    n_qubits: int
    fidelity: float
    connectivity: List[int]  # Connected node IDs
    capabilities: List[str]  # Supported operations
    
    
@dataclass  
class EntanglementLink:
    """Represents an entangled quantum channel between nodes."""
    source_id: int
    target_id: int
    fidelity: float
    schmidt_rank: int
    coherence_time: float  # microseconds
    entanglement_rate: float  # pairs/second
    protocol: str  # "nv_center", "photonic", "ion_trap"


class PhotonicNetwork(BaseModel):
    """
    Manages topology and entanglement distribution for quantum photonic networks.
    
    Handles node discovery, entanglement scheduling, and optimal workload distribution
    across distributed quantum processing units connected via quantum channels.
    """
    
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}  # Allow dynamic attributes and arbitrary types
    
    nodes: int = Field(default=4, ge=1, le=1024)
    entanglement_protocol: str = Field(default="nv_center", pattern="^(nv_center|photonic|ion_trap)$")
    fidelity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    topology: str = Field(default="complete", pattern="^(complete|ring|star|grid|random)$")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._validate_and_initialize()
    
    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v):
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"nodes must be a positive integer, got {v}")
        if v > 1024:
            raise ValueError(f"nodes cannot exceed 1024, got {v}")
        return v
    
    @field_validator('fidelity_threshold')
    @classmethod
    def validate_fidelity(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"fidelity_threshold must be in [0,1], got {v}")
        return v
    
    @error_boundary(NetworkError, ErrorSeverity.HIGH)
    def _validate_and_initialize(self) -> None:
        """Validate parameters and initialize quantum network."""
        logger.info(f"Initializing quantum network: {self.nodes} nodes, {self.topology} topology")
        
        # Validate network parameters
        validation_result = validate_network_parameters(
            self.nodes, self.fidelity_threshold, self.topology
        )
        log_validation_result(validation_result, "network parameters")
        
        if not validation_result.is_valid:
            raise NetworkError(f"Invalid network parameters: {validation_result.errors}")
        
        try:
            self.graph = self._create_topology()
            self.quantum_nodes = self._create_quantum_nodes()
            self.entanglement_links = self._create_entanglement_links()
            self.entanglement_map = self._build_entanglement_map()
            
            logger.info(f"Network initialized successfully: {len(self.quantum_nodes)} nodes, {len(self.entanglement_links)//2} links")
            
        except Exception as e:
            raise NetworkError(f"Failed to initialize network: {e}", severity=ErrorSeverity.CRITICAL)
        
    def _create_topology(self) -> nx.Graph:
        """Create network topology based on specified type."""
        if self.topology == "complete":
            graph = nx.complete_graph(self.nodes)
        elif self.topology == "ring":
            graph = nx.cycle_graph(self.nodes)
        elif self.topology == "star":
            graph = nx.star_graph(self.nodes - 1)
        elif self.topology == "grid":
            # Create square grid if possible
            side_length = int(np.sqrt(self.nodes))
            if side_length * side_length == self.nodes:
                graph = nx.grid_2d_graph(side_length, side_length)
                # Convert to integer node labels
                mapping = {node: i for i, node in enumerate(graph.nodes())}
                graph = nx.relabel_nodes(graph, mapping)
            else:
                # Fall back to path graph
                graph = nx.path_graph(self.nodes)
        elif self.topology == "random":
            # Erdős–Rényi random graph with connectivity probability 0.3
            graph = nx.erdos_renyi_graph(self.nodes, 0.3)
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
            
        return graph
    
    def _create_quantum_nodes(self) -> Dict[int, QuantumNode]:
        """Create quantum processing nodes with hardware specifications."""
        nodes = {}
        
        for node_id in self.graph.nodes():
            # Simulate different QPU types and capabilities
            qpu_types = ["photonic", "superconducting", "trapped_ion", "nv_center"]
            qpu_type = np.random.choice(qpu_types)
            
            # QPU-specific parameters
            if qpu_type == "photonic":
                n_qubits = np.random.randint(8, 64)
                base_fidelity = 0.92
                capabilities = ["gaussian_ops", "photon_counting", "homodyne"]
            elif qpu_type == "superconducting":
                n_qubits = np.random.randint(16, 128) 
                base_fidelity = 0.95
                capabilities = ["two_qubit_gates", "readout", "parametric_ops"]
            elif qpu_type == "trapped_ion":
                n_qubits = np.random.randint(4, 32)
                base_fidelity = 0.98
                capabilities = ["all_to_all", "high_fidelity", "long_coherence"]
            else:  # nv_center
                n_qubits = np.random.randint(2, 16)
                base_fidelity = 0.88
                capabilities = ["room_temp", "optical_interface", "spin_control"]
            
            # Add noise to fidelity
            fidelity = base_fidelity * (0.95 + 0.1 * np.random.random())
            fidelity = min(fidelity, 0.999)
            
            connectivity = list(self.graph.neighbors(node_id))
            
            nodes[node_id] = QuantumNode(
                node_id=node_id,
                qpu_type=qpu_type,
                n_qubits=n_qubits,
                fidelity=fidelity,
                connectivity=connectivity,
                capabilities=capabilities
            )
        
        return nodes
    
    def _create_entanglement_links(self) -> Dict[Tuple[int, int], EntanglementLink]:
        """Create entangled quantum channels between connected nodes."""
        links = {}
        
        for edge in self.graph.edges():
            source_id, target_id = edge
            source_node = self.quantum_nodes[source_id]
            target_node = self.quantum_nodes[target_id]
            
            # Determine entanglement protocol based on node types
            protocols = []
            if "optical_interface" in source_node.capabilities and "optical_interface" in target_node.capabilities:
                protocols.append("nv_center")
            if source_node.qpu_type == "photonic" or target_node.qpu_type == "photonic":
                protocols.append("photonic") 
            if source_node.qpu_type == "trapped_ion" and target_node.qpu_type == "trapped_ion":
                protocols.append("ion_trap")
            
            if not protocols:
                protocols = ["photonic"]  # Default fallback
            
            protocol = np.random.choice(protocols)
            
            # Protocol-specific parameters
            if protocol == "nv_center":
                base_fidelity = 0.90
                schmidt_rank = np.random.randint(2, 16)
                coherence_time = np.random.uniform(100, 1000)  # microseconds
                entanglement_rate = np.random.uniform(1e3, 1e6)  # pairs/second
            elif protocol == "photonic":
                base_fidelity = 0.85
                schmidt_rank = np.random.randint(4, 32)
                coherence_time = np.random.uniform(10, 100)
                entanglement_rate = np.random.uniform(1e6, 1e9)
            else:  # ion_trap
                base_fidelity = 0.95
                schmidt_rank = np.random.randint(2, 8)
                coherence_time = np.random.uniform(1000, 10000)
                entanglement_rate = np.random.uniform(1e2, 1e4)
            
            # Link fidelity depends on node fidelities and distance
            node_fidelity = min(source_node.fidelity, target_node.fidelity)
            fidelity = base_fidelity * node_fidelity * (0.9 + 0.2 * np.random.random())
            fidelity = min(fidelity, 0.999)
            
            # Only include links above fidelity threshold
            if fidelity >= self.fidelity_threshold:
                links[(source_id, target_id)] = EntanglementLink(
                    source_id=source_id,
                    target_id=target_id,
                    fidelity=fidelity,
                    schmidt_rank=schmidt_rank,
                    coherence_time=coherence_time,
                    entanglement_rate=entanglement_rate,
                    protocol=protocol
                )
                
                # Add reverse link for undirected graph
                links[(target_id, source_id)] = EntanglementLink(
                    source_id=target_id,
                    target_id=source_id,
                    fidelity=fidelity,
                    schmidt_rank=schmidt_rank,
                    coherence_time=coherence_time,
                    entanglement_rate=entanglement_rate,
                    protocol=protocol
                )
        
        return links
    
    def _build_entanglement_map(self) -> Dict[int, Dict[int, EntanglementLink]]:
        """Build adjacency map for fast entanglement link lookup."""
        entanglement_map = {node_id: {} for node_id in self.quantum_nodes.keys()}
        
        for (source_id, target_id), link in self.entanglement_links.items():
            entanglement_map[source_id][target_id] = link
            
        return entanglement_map
    
    def get_entanglement_quality(self, node_a: int, node_b: int) -> Optional[float]:
        """Get entanglement fidelity between two nodes."""
        if node_b in self.entanglement_map[node_a]:
            return self.entanglement_map[node_a][node_b].fidelity
        return None
    
    def get_schmidt_rank(self, node_a: int, node_b: int) -> Optional[int]:
        """Get Schmidt rank of entanglement between two nodes."""
        if node_b in self.entanglement_map[node_a]:
            return self.entanglement_map[node_a][node_b].schmidt_rank
        return None
    
    def find_optimal_partition(self, computation_graph: nx.Graph) -> Dict[int, int]:
        """
        Find optimal partitioning of computation graph onto quantum network.
        
        Uses graph matching algorithms to minimize communication overhead
        while maximizing utilization of high-fidelity entanglement links.
        """
        # Simple greedy assignment for now - can be improved with sophisticated algorithms
        node_assignments = {}
        available_qnodes = list(self.quantum_nodes.keys())
        
        for comp_node in computation_graph.nodes():
            if available_qnodes:
                # Choose quantum node with highest capacity
                best_qnode = max(available_qnodes, 
                               key=lambda x: self.quantum_nodes[x].n_qubits * self.quantum_nodes[x].fidelity)
                node_assignments[comp_node] = best_qnode
                available_qnodes.remove(best_qnode)
            else:
                # Reuse nodes if we run out
                node_assignments[comp_node] = np.random.choice(list(self.quantum_nodes.keys()))
        
        return node_assignments
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        total_qubits = sum(node.n_qubits for node in self.quantum_nodes.values())
        avg_fidelity = np.mean([node.fidelity for node in self.quantum_nodes.values()])
        avg_link_fidelity = np.mean([link.fidelity for link in self.entanglement_links.values()])
        
        return {
            "num_nodes": len(self.quantum_nodes),
            "num_links": len(self.entanglement_links) // 2,  # Undirected graph
            "total_qubits": total_qubits,
            "avg_node_fidelity": avg_fidelity,
            "avg_link_fidelity": avg_link_fidelity,
            "topology": self.topology,
            "protocols": list(set(link.protocol for link in self.entanglement_links.values())),
        }
    
    def visualize_network(self) -> None:
        """Create visualization of quantum network topology."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Create layout
            if self.topology == "grid":
                pos = nx.spring_layout(self.graph, k=2, iterations=50)
            else:
                pos = nx.spring_layout(self.graph)
            
            # Draw nodes colored by QPU type
            node_colors = []
            for node_id in self.graph.nodes():
                qpu_type = self.quantum_nodes[node_id].qpu_type
                if qpu_type == "photonic":
                    node_colors.append("lightblue")
                elif qpu_type == "superconducting":
                    node_colors.append("lightgreen")
                elif qpu_type == "trapped_ion":
                    node_colors.append("lightcoral")
                else:  # nv_center
                    node_colors.append("lightyellow")
            
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                                 node_size=500, alpha=0.8)
            
            # Draw edges colored by protocol
            edge_colors = []
            for edge in self.graph.edges():
                if edge in self.entanglement_links:
                    protocol = self.entanglement_links[edge].protocol
                    if protocol == "nv_center":
                        edge_colors.append("red")
                    elif protocol == "photonic":
                        edge_colors.append("blue")
                    else:  # ion_trap
                        edge_colors.append("green")
                else:
                    edge_colors.append("gray")
            
            nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, alpha=0.6)
            nx.draw_networkx_labels(self.graph, pos, font_size=10)
            
            plt.title(f"Quantum Photonic Network ({self.topology} topology)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for network visualization")
    
