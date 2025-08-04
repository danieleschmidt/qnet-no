"""Entanglement-aware task scheduling and resource allocation."""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from enum import Enum
import heapq
from .photonic_network import PhotonicNetwork, QuantumNode, EntanglementLink


class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ComputationTask:
    """Represents a quantum computation task."""
    task_id: str
    operation_type: str  # "fourier_transform", "tensor_contraction", "gate_sequence"
    required_qubits: int
    estimated_time: float  # microseconds
    priority: TaskPriority
    dependencies: Set[str] = field(default_factory=set)
    preferred_nodes: Optional[List[int]] = None
    deadline: Optional[float] = None
    quantum_volume: int = 1  # Measure of computational complexity
    

@dataclass
class SchedulingResult:
    """Result of task scheduling with resource allocation."""
    task_assignments: Dict[str, int]  # task_id -> node_id
    execution_order: List[str]
    estimated_completion_time: float
    resource_utilization: Dict[int, float]
    entanglement_usage: Dict[Tuple[int, int], float]
    

class EntanglementScheduler:
    """
    Optimal scheduling of quantum tasks across distributed photonic network.
    
    Implements entanglement-aware algorithms that consider quantum resource
    constraints, decoherence times, and inter-node communication overhead.
    """
    
    def __init__(self, network: PhotonicNetwork):
        self.network = network
        self.current_time = 0.0
        self.node_availability = {node_id: 0.0 for node_id in network.quantum_nodes.keys()}
        self.task_queue = []
        self.completed_tasks = set()
        self.running_tasks = {}
        
    def submit_task(self, task: ComputationTask) -> None:
        """Submit a task to the scheduling queue."""
        # Priority queue with task priority and submission time
        priority_value = -task.priority.value  # Negative for max-heap behavior
        heapq.heappush(self.task_queue, (priority_value, self.current_time, task))
    
    def schedule_tasks(self, tasks: List[ComputationTask]) -> SchedulingResult:
        """
        Schedule multiple tasks optimally across quantum network.
        
        Uses hybrid classical-quantum optimization considering:
        - Entanglement fidelity and coherence times
        - Node capabilities and current load
        - Task dependencies and deadlines
        - Communication overhead between nodes
        """
        # Submit all tasks to queue
        for task in tasks:
            self.submit_task(task)
        
        task_assignments = {}
        execution_order = []
        resource_utilization = {node_id: 0.0 for node_id in self.network.quantum_nodes.keys()}
        entanglement_usage = {}
        
        # Main scheduling loop
        while self.task_queue or self.running_tasks:
            # Check for completed tasks
            self._update_completed_tasks()
            
            # Schedule next ready task
            if self.task_queue:
                ready_tasks = self._get_ready_tasks()
                
                for task in ready_tasks:
                    optimal_node = self._find_optimal_node(task)
                    
                    if optimal_node is not None:
                        # Assign task to node
                        task_assignments[task.task_id] = optimal_node
                        execution_order.append(task.task_id)
                        
                        # Update resource utilization
                        execution_time = self._estimate_execution_time(task, optimal_node)
                        start_time = max(self.current_time, self.node_availability[optimal_node])
                        end_time = start_time + execution_time
                        
                        self.node_availability[optimal_node] = end_time
                        self.running_tasks[task.task_id] = {
                            'task': task,
                            'node': optimal_node,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                        
                        # Track resource utilization
                        resource_utilization[optimal_node] += execution_time
                        
                        # Track entanglement usage for distributed tasks
                        entanglement_links = self._get_required_entanglement(task, optimal_node)
                        for link in entanglement_links:
                            if link not in entanglement_usage:
                                entanglement_usage[link] = 0
                            entanglement_usage[link] += execution_time
            
            # Advance time to next event
            self._advance_time()
        
        # Calculate final metrics
        total_time = max(self.node_availability.values()) if self.node_availability else 0
        
        # Normalize resource utilization
        for node_id in resource_utilization:
            if total_time > 0:
                resource_utilization[node_id] /= total_time
        
        return SchedulingResult(
            task_assignments=task_assignments,
            execution_order=execution_order,
            estimated_completion_time=total_time,
            resource_utilization=resource_utilization,
            entanglement_usage=entanglement_usage
        )
    
    def _get_ready_tasks(self) -> List[ComputationTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        remaining_tasks = []
        
        while self.task_queue:
            priority, submit_time, task = heapq.heappop(self.task_queue)
            
            # Check if dependencies are satisfied
            if task.dependencies.issubset(self.completed_tasks):
                ready_tasks.append(task)
            else:
                remaining_tasks.append((priority, submit_time, task))
        
        # Put non-ready tasks back in queue
        for item in remaining_tasks:
            heapq.heappush(self.task_queue, item)
        
        return ready_tasks
    
    def _find_optimal_node(self, task: ComputationTask) -> Optional[int]:
        """
        Find optimal node for task execution using multi-objective optimization.
        
        Considers:
        - Node capabilities and qubit count
        - Current load and availability
        - Entanglement quality to other nodes
        - Communication overhead
        """
        candidate_nodes = []
        
        # Filter nodes by capability and capacity
        for node_id, node in self.network.quantum_nodes.items():
            if (task.required_qubits <= node.n_qubits and 
                self._check_node_capability(task, node)):
                
                candidate_nodes.append(node_id)
        
        if not candidate_nodes:
            return None
        
        # Multi-objective scoring
        best_node = None
        best_score = -np.inf
        
        for node_id in candidate_nodes:
            score = self._calculate_node_score(task, node_id)
            
            if score > best_score:
                best_score = score
                best_node = node_id
        
        return best_node
    
    def _calculate_node_score(self, task: ComputationTask, node_id: int) -> float:
        """Calculate multi-objective score for assigning task to node."""
        node = self.network.quantum_nodes[node_id]
        
        # Capability score (higher is better)
        capability_score = node.fidelity * (node.n_qubits / task.required_qubits)
        
        # Availability score (earlier availability is better)
        current_load = self.node_availability[node_id] - self.current_time
        availability_score = 1.0 / (1.0 + current_load / 1000.0)  # Normalize by 1ms
        
        # Entanglement quality score
        entanglement_score = self._calculate_entanglement_score(task, node_id)
        
        # Communication overhead score
        communication_score = self._calculate_communication_score(task, node_id)
        
        # Preference score
        preference_score = 1.0
        if task.preferred_nodes and node_id in task.preferred_nodes:
            preference_score = 1.5
        
        # Weighted combination
        total_score = (
            0.3 * capability_score +
            0.2 * availability_score +
            0.2 * entanglement_score +
            0.2 * communication_score +
            0.1 * preference_score
        )
        
        return total_score
    
    def _calculate_entanglement_score(self, task: ComputationTask, node_id: int) -> float:
        """Calculate score based on entanglement quality to other nodes."""
        if task.operation_type in ["local_gate", "measurement"]:
            return 1.0  # No entanglement needed
        
        # Average entanglement quality to all other nodes
        entanglement_qualities = []
        for other_node in self.network.quantum_nodes.keys():
            if other_node != node_id:
                quality = self.network.get_entanglement_quality(node_id, other_node)
                if quality is not None:
                    entanglement_qualities.append(quality)
        
        if entanglement_qualities:
            return np.mean(entanglement_qualities)
        else:
            return 0.1  # Poor score if no entanglement available
    
    def _calculate_communication_score(self, task: ComputationTask, node_id: int) -> float:
        """Calculate score based on communication overhead."""
        if task.operation_type in ["local_gate", "measurement"]:
            return 1.0
        
        # Simple model: communication overhead increases with network distance
        avg_distance = self._calculate_average_distance(node_id)
        return 1.0 / (1.0 + avg_distance / 10.0)
    
    def _calculate_average_distance(self, node_id: int) -> float:
        """Calculate average shortest path distance to other nodes."""
        distances = []
        
        for other_node in self.network.quantum_nodes.keys():
            if other_node != node_id:
                try:
                    distance = nx.shortest_path_length(self.network.graph, node_id, other_node)
                    distances.append(distance)
                except nx.NetworkXNoPath:
                    distances.append(float('inf'))
        
        finite_distances = [d for d in distances if d != float('inf')]
        return np.mean(finite_distances) if finite_distances else float('inf')
    
    def _check_node_capability(self, task: ComputationTask, node: QuantumNode) -> bool:
        """Check if node has required capabilities for task."""
        capability_map = {
            "fourier_transform": ["gaussian_ops", "photon_counting"],
            "tensor_contraction": ["two_qubit_gates", "parametric_ops"],
            "gate_sequence": ["two_qubit_gates", "readout"],
            "measurement": ["readout", "photon_counting"]
        }
        
        required_caps = capability_map.get(task.operation_type, [])
        return any(cap in node.capabilities for cap in required_caps) or not required_caps
    
    def _estimate_execution_time(self, task: ComputationTask, node_id: int) -> float:
        """Estimate task execution time on specific node."""
        node = self.network.quantum_nodes[node_id]
        
        # Base time from task estimate
        base_time = task.estimated_time
        
        # Scale by node efficiency (higher fidelity = faster execution)
        efficiency_factor = node.fidelity
        
        # Scale by quantum volume (more complex tasks take longer)
        complexity_factor = task.quantum_volume ** 0.5
        
        # Communication overhead for distributed tasks
        comm_overhead = self._calculate_communication_overhead(task, node_id)
        
        total_time = (base_time / efficiency_factor) * complexity_factor + comm_overhead
        return total_time
    
    def _calculate_communication_overhead(self, task: ComputationTask, node_id: int) -> float:
        """Calculate communication overhead for distributed operations."""
        if task.operation_type in ["local_gate", "measurement"]:
            return 0.0
        
        # Simple model: overhead proportional to number of required entangled links
        required_links = len(self._get_required_entanglement(task, node_id))
        return required_links * 10.0  # 10 microseconds per link
    
    def _get_required_entanglement(self, task: ComputationTask, node_id: int) -> List[Tuple[int, int]]:
        """Get list of entanglement links required for task execution."""
        if task.operation_type in ["local_gate", "measurement"]:
            return []
        
        # For distributed operations, assume entanglement with all neighbors
        links = []
        for neighbor in self.network.graph.neighbors(node_id):
            links.append((node_id, neighbor))
        
        return links
    
    def _update_completed_tasks(self) -> None:
        """Update completed tasks and free resources."""
        completed = []
        
        for task_id, task_info in self.running_tasks.items():
            if self.current_time >= task_info['end_time']:
                completed.append(task_id)
                self.completed_tasks.add(task_id)
        
        for task_id in completed:
            del self.running_tasks[task_id]
    
    def _advance_time(self) -> None:
        """Advance simulation time to next event."""
        if self.running_tasks:
            next_completion = min(info['end_time'] for info in self.running_tasks.values())
            self.current_time = next_completion
        else:
            self.current_time += 1.0  # Small time step if no running tasks
    
    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduling performance metrics."""
        total_nodes = len(self.network.quantum_nodes)
        active_nodes = sum(1 for util in self.node_availability.values() if util > 0)
        
        return {
            "total_tasks_completed": len(self.completed_tasks),
            "total_execution_time": max(self.node_availability.values()) if self.node_availability else 0,
            "node_utilization": active_nodes / total_nodes if total_nodes > 0 else 0,
            "average_task_wait_time": self._calculate_average_wait_time(),
            "entanglement_efficiency": self._calculate_entanglement_efficiency(),
        }
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average task waiting time."""
        if not self.running_tasks:
            return 0.0
        
        wait_times = []
        for task_info in self.running_tasks.values():
            wait_time = task_info['start_time'] - task_info.get('submit_time', 0)
            wait_times.append(max(0, wait_time))  # Non-negative wait times
        
        return np.mean(wait_times) if wait_times else 0.0
    
    def _calculate_entanglement_efficiency(self) -> float:
        """Calculate efficiency of entanglement resource usage."""
        total_links = len(self.network.entanglement_links) // 2  # Undirected
        if total_links == 0:
            return 0.0
        
        # Simple metric: fraction of links that were used
        used_links = set()
        for task_info in self.running_tasks.values():
            required_links = self._get_required_entanglement(task_info['task'], task_info['node'])
            used_links.update(required_links)
        
        return len(used_links) / total_links