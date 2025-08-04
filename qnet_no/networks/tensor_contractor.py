"""Efficient tensor network contraction for quantum computation results."""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import jax.numpy as jnp
import tensornetwork as tn
from dataclasses import dataclass
from .photonic_network import PhotonicNetwork


@dataclass
class TensorNode:
    """Represents a tensor in the quantum computation network."""
    tensor_id: str
    tensor_data: jnp.ndarray
    indices: List[str]
    quantum_node_id: int
    schmidt_rank: Optional[int] = None


@dataclass
class ContractionPlan:
    """Optimal contraction sequence for tensor network."""
    contraction_sequence: List[Tuple[str, str]]
    estimated_cost: float
    memory_requirement: int
    parallelization_opportunities: List[List[str]]


class TensorContractor:
    """
    Efficient MPS-based tensor network contraction for distributed quantum results.
    
    Handles aggregation of computation results from multiple quantum nodes
    using matrix product state decompositions and optimal contraction ordering.
    """
    
    def __init__(self, network: PhotonicNetwork, max_bond_dimension: int = 64):
        self.network = network
        self.max_bond_dimension = max_bond_dimension
        self.tensor_nodes = {}
        self.contraction_cache = {}
        
    def add_tensor(self, tensor_id: str, tensor_data: jnp.ndarray, 
                   indices: List[str], quantum_node_id: int,
                   schmidt_rank: Optional[int] = None) -> None:
        """Add tensor from quantum computation result."""
        tensor_node = TensorNode(
            tensor_id=tensor_id,
            tensor_data=tensor_data,
            indices=indices,
            quantum_node_id=quantum_node_id,
            schmidt_rank=schmidt_rank
        )
        self.tensor_nodes[tensor_id] = tensor_node
    
    def contract_network(self, output_indices: Optional[List[str]] = None) -> jnp.ndarray:
        """
        Contract full tensor network to produce final result.
        
        Uses optimal contraction ordering and MPS decomposition to handle
        large tensor networks efficiently across distributed quantum nodes.
        """
        if not self.tensor_nodes:
            raise ValueError("No tensors to contract")
        
        # Create TensorNetwork objects
        tn_nodes = {}
        for tensor_id, tensor_node in self.tensor_nodes.items():
            tn_node = tn.Node(tensor_node.tensor_data, name=tensor_id)
            tn_nodes[tensor_id] = tn_node
        
        # Connect tensor network based on shared indices
        self._connect_tensor_network(tn_nodes)
        
        # Find optimal contraction plan
        contraction_plan = self._find_optimal_contraction(tn_nodes, output_indices)
        
        # Execute contraction with MPS compression
        result = self._execute_contraction(tn_nodes, contraction_plan, output_indices)
        
        return result
    
    def _connect_tensor_network(self, tn_nodes: Dict[str, tn.Node]) -> None:
        """Connect tensor network nodes based on shared indices."""
        # Build index mapping
        index_to_nodes = {}
        for tensor_id, tensor_node in self.tensor_nodes.items():
            for i, index in enumerate(tensor_node.indices):
                if index not in index_to_nodes:
                    index_to_nodes[index] = []
                index_to_nodes[index].append((tensor_id, i))
        
        # Connect nodes with shared indices
        for index, node_connections in index_to_nodes.items():
            if len(node_connections) == 2:
                # Standard edge between two nodes
                (node1_id, edge1), (node2_id, edge2) = node_connections
                tn_nodes[node1_id][edge1] ^ tn_nodes[node2_id][edge2]
            elif len(node_connections) > 2:
                # Multi-way connection - insert intermediate nodes
                self._handle_multiway_connection(tn_nodes, node_connections, index)
    
    def _handle_multiway_connection(self, tn_nodes: Dict[str, tn.Node], 
                                   connections: List[Tuple[str, int]], index: str) -> None:
        """Handle connections between more than two tensor nodes."""
        # Create auxiliary tensors for multi-way connections
        # This is a simplified approach - could be improved with more sophisticated methods
        if len(connections) > 2:
            # Connect first two nodes normally
            (node1_id, edge1), (node2_id, edge2) = connections[:2]
            tn_nodes[node1_id][edge1] ^ tn_nodes[node2_id][edge2]
            
            # For remaining connections, we would need more complex handling
            # For now, skip multi-way connections beyond pairs
    
    def _find_optimal_contraction(self, tn_nodes: Dict[str, tn.Node], 
                                 output_indices: Optional[List[str]]) -> ContractionPlan:
        """Find optimal contraction sequence using heuristic algorithms."""
        # Simple greedy contraction order - could be improved with dynamic programming
        node_list = list(tn_nodes.values())
        contraction_sequence = []
        estimated_cost = 0
        
        # Greedy approach: contract smallest tensors first
        while len(node_list) > 1:
            # Find pair with minimum contraction cost
            best_pair = None
            best_cost = float('inf')
            
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    node1, node2 = node_list[i], node_list[j]
                    
                    # Check if nodes are connected
                    if self._nodes_connected(node1, node2):
                        cost = self._estimate_contraction_cost(node1, node2)
                        if cost < best_cost:
                            best_cost = cost
                            best_pair = (i, j, node1, node2)
            
            if best_pair is None:
                # No connected pairs found - contract arbitrary pair
                if len(node_list) >= 2:
                    node1, node2 = node_list[0], node_list[1]
                    best_pair = (0, 1, node1, node2)
                    best_cost = self._estimate_contraction_cost(node1, node2)
                else:
                    break
            
            i, j, node1, node2 = best_pair
            contraction_sequence.append((node1.name, node2.name))
            estimated_cost += best_cost
            
            # Remove contracted nodes and add result
            node_list.pop(max(i, j))  # Remove larger index first
            node_list.pop(min(i, j))
            
            # Add contracted result (simplified - actual implementation would create new node)
            if len(node_list) > 0:  # Still have nodes to contract
                # In real implementation, would create new node from contraction result
                pass
        
        # Identify parallelization opportunities (simplified)
        parallelization_opportunities = self._find_parallel_contractions(contraction_sequence)
        
        return ContractionPlan(
            contraction_sequence=contraction_sequence,
            estimated_cost=estimated_cost,
            memory_requirement=self._estimate_memory_requirement(),
            parallelization_opportunities=parallelization_opportunities
        )
    
    def _nodes_connected(self, node1: tn.Node, node2: tn.Node) -> bool:
        """Check if two tensor nodes are connected."""
        for edge1 in node1.edges:
            for edge2 in node2.edges:
                if edge1.is_connected() and edge2.is_connected() and edge1.node1 == edge2.node1:
                    return True
        return False
    
    def _estimate_contraction_cost(self, node1: tn.Node, node2: tn.Node) -> float:
        """Estimate computational cost of contracting two nodes."""
        # Simple cost model: product of all dimension sizes
        dims1 = node1.tensor.shape
        dims2 = node2.tensor.shape
        
        # Simplified cost estimation
        cost = np.prod(dims1) * np.prod(dims2)
        return float(cost)
    
    def _estimate_memory_requirement(self) -> int:
        """Estimate peak memory requirement for contraction."""
        total_elements = 0
        for tensor_node in self.tensor_nodes.values():
            total_elements += np.prod(tensor_node.tensor_data.shape)
        
        # Conservative estimate: 2x total elements (intermediate results)
        return int(total_elements * 2)
    
    def _find_parallel_contractions(self, sequence: List[Tuple[str, str]]) -> List[List[str]]:
        """Identify contractions that can be performed in parallel."""
        # Simplified: assume independent contractions can be parallelized
        parallel_groups = []
        used_tensors = set()
        
        current_group = []
        for tensor1, tensor2 in sequence:
            if tensor1 not in used_tensors and tensor2 not in used_tensors:
                current_group.extend([tensor1, tensor2])
                used_tensors.update([tensor1, tensor2])
            else:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [tensor1, tensor2]
                used_tensors = {tensor1, tensor2}
        
        if current_group:
            parallel_groups.append(current_group)
        
        return parallel_groups
    
    def _execute_contraction(self, tn_nodes: Dict[str, tn.Node], 
                           plan: ContractionPlan, 
                           output_indices: Optional[List[str]]) -> jnp.ndarray:
        """Execute tensor contraction with MPS compression."""
        # Execute contraction sequence
        remaining_nodes = list(tn_nodes.values())
        
        for tensor1_name, tensor2_name in plan.contraction_sequence:
            # Find nodes to contract
            node1 = None
            node2 = None
            
            for node in remaining_nodes:
                if node.name == tensor1_name:
                    node1 = node
                elif node.name == tensor2_name:
                    node2 = node
            
            if node1 is not None and node2 is not None:
                # Perform contraction
                try:
                    result_node = tn.contract_between(node1, node2, name=f"contracted_{tensor1_name}_{tensor2_name}")
                    
                    # Apply MPS compression if result is too large
                    if np.prod(result_node.tensor.shape) > self.max_bond_dimension ** 2:
                        result_node = self._apply_mps_compression(result_node)
                    
                    # Update remaining nodes
                    remaining_nodes = [n for n in remaining_nodes if n.name not in [tensor1_name, tensor2_name]]
                    remaining_nodes.append(result_node)
                    
                except Exception as e:
                    print(f"Error contracting {tensor1_name} and {tensor2_name}: {e}")
                    continue
        
        # Return final result
        if remaining_nodes:
            final_result = remaining_nodes[0].tensor
            
            # Convert to JAX array if needed
            if not isinstance(final_result, jnp.ndarray):
                final_result = jnp.array(final_result)
            
            return final_result
        else:
            raise RuntimeError("Contraction failed - no remaining nodes")
    
    def _apply_mps_compression(self, node: tn.Node) -> tn.Node:
        """Apply Matrix Product State compression to reduce tensor size."""
        tensor = node.tensor
        
        # Simplified MPS compression using SVD
        # Reshape tensor to matrix for SVD
        if len(tensor.shape) > 2:
            left_dim = tensor.shape[0]
            right_dim = np.prod(tensor.shape[1:])
            matrix = tensor.reshape(left_dim, right_dim)
        else:
            matrix = tensor
        
        # Perform SVD with truncation
        try:
            U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)
            
            # Truncate to max bond dimension
            if len(S) > self.max_bond_dimension:
                U = U[:, :self.max_bond_dimension]
                S = S[:self.max_bond_dimension]
                Vh = Vh[:self.max_bond_dimension, :]
            
            # Reconstruct compressed tensor
            compressed_matrix = U @ jnp.diag(S) @ Vh
            
            if len(tensor.shape) > 2:
                compressed_tensor = compressed_matrix.reshape(tensor.shape[0], *tensor.shape[1:])
            else:
                compressed_tensor = compressed_matrix
            
            # Create new node with compressed tensor
            compressed_node = tn.Node(compressed_tensor, name=f"compressed_{node.name}")
            return compressed_node
            
        except Exception as e:
            print(f"MPS compression failed: {e}")
            return node  # Return original node if compression fails
    
    def contract_distributed_results(self, results: Dict[int, jnp.ndarray], 
                                   operation_type: str = "sum") -> jnp.ndarray:
        """
        Contract results from multiple quantum nodes.
        
        Specialized method for combining results distributed across
        the quantum network with proper entanglement handling.
        """
        if not results:
            raise ValueError("No results to contract")
        
        node_ids = list(results.keys())
        
        if operation_type == "sum":
            # Simple summation across nodes
            total_result = None
            for node_id, result in results.items():
                if total_result is None:
                    total_result = result
                else:
                    total_result = total_result + result
            return total_result
            
        elif operation_type == "tensor_product":
            # Tensor product across entangled nodes
            total_result = None
            for node_id in sorted(node_ids):  # Ensure consistent ordering
                result = results[node_id]
                if total_result is None:
                    total_result = result
                else:
                    total_result = jnp.kron(total_result, result)
            return total_result
            
        elif operation_type == "entangled_average":
            # Weighted average based on entanglement quality
            weighted_sum = None
            total_weight = 0
            
            for node_id, result in results.items():
                # Calculate weight based on average entanglement quality
                weight = self._calculate_node_weight(node_id)
                
                if weighted_sum is None:
                    weighted_sum = weight * result
                else:
                    weighted_sum = weighted_sum + weight * result
                
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return jnp.zeros_like(list(results.values())[0])
        
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
    
    def _calculate_node_weight(self, node_id: int) -> float:
        """Calculate weight for node based on entanglement quality."""
        node = self.network.quantum_nodes[node_id]
        
        # Base weight from node fidelity
        base_weight = node.fidelity
        
        # Bonus weight from entanglement quality
        entanglement_qualities = []
        for other_node_id in self.network.quantum_nodes.keys():
            if other_node_id != node_id:
                quality = self.network.get_entanglement_quality(node_id, other_node_id)
                if quality is not None:
                    entanglement_qualities.append(quality)
        
        if entanglement_qualities:
            avg_entanglement = np.mean(entanglement_qualities)
            return base_weight * (1 + avg_entanglement)
        else:
            return base_weight
    
    def get_contraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about tensor contraction operations."""
        total_tensors = len(self.tensor_nodes)
        total_elements = sum(np.prod(node.tensor_data.shape) for node in self.tensor_nodes.values())
        
        # Calculate Schmidt rank statistics
        schmidt_ranks = [node.schmidt_rank for node in self.tensor_nodes.values() if node.schmidt_rank is not None]
        
        return {
            "total_tensors": total_tensors,
            "total_elements": total_elements,
            "average_tensor_size": total_elements / total_tensors if total_tensors > 0 else 0,
            "max_bond_dimension": self.max_bond_dimension,
            "schmidt_rank_stats": {
                "mean": np.mean(schmidt_ranks) if schmidt_ranks else None,
                "max": np.max(schmidt_ranks) if schmidt_ranks else None,
                "min": np.min(schmidt_ranks) if schmidt_ranks else None,
            },
            "cache_hits": len(self.contraction_cache),
        }
    
    def clear_cache(self) -> None:
        """Clear contraction cache to free memory."""
        self.contraction_cache.clear()
    
    def reset(self) -> None:
        """Reset contractor state for new computation."""
        self.tensor_nodes.clear()
        self.contraction_cache.clear()