"""Distributed computing utilities for quantum neural operators."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import logging
from queue import Queue, Empty
import socket
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ComputeNode:
    """Represents a compute node in distributed system."""
    node_id: str
    host: str
    port: int
    capabilities: List[str]
    load: float = 0.0
    last_heartbeat: float = 0.0
    status: str = "idle"  # idle, busy, failed


@dataclass
class DistributedTask:
    """A task to be executed on distributed compute nodes."""
    task_id: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: int = 0
    dependencies: List[str] = None
    node_requirements: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class LoadBalancer:
    """Load balancer for distributing quantum computations."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.nodes: Dict[str, ComputeNode] = {}
        self.current_node_index = 0
        self.lock = threading.Lock()
    
    def register_node(self, node: ComputeNode):
        """Register a compute node."""
        with self.lock:
            self.nodes[node.node_id] = node
            logger.info(f"Registered compute node: {node.node_id} at {node.host}:{node.port}")
    
    def unregister_node(self, node_id: str):
        """Unregister a compute node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered compute node: {node_id}")
    
    def select_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select best node for task execution."""
        with self.lock:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == "idle" and self._node_meets_requirements(node, task)
            ]
            
            if not available_nodes:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(available_nodes)
            elif self.strategy == "least_loaded":
                return self._least_loaded_selection(available_nodes)
            elif self.strategy == "capability_based":
                return self._capability_based_selection(available_nodes, task)
            else:
                return available_nodes[0]
    
    def _node_meets_requirements(self, node: ComputeNode, task: DistributedTask) -> bool:
        """Check if node meets task requirements."""
        if task.node_requirements is None:
            return True
        
        return all(req in node.capabilities for req in task.node_requirements)
    
    def _round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node using round-robin strategy."""
        self.current_node_index = (self.current_node_index + 1) % len(nodes)
        return nodes[self.current_node_index]
    
    def _least_loaded_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select least loaded node."""
        return min(nodes, key=lambda n: n.load)
    
    def _capability_based_selection(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node based on capabilities match."""
        if task.node_requirements is None:
            return self._least_loaded_selection(nodes)
        
        # Score nodes based on capability match
        scored_nodes = []
        for node in nodes:
            score = len(set(node.capabilities) & set(task.node_requirements))
            scored_nodes.append((score, node.load, node))
        
        # Sort by score (descending) then by load (ascending)
        scored_nodes.sort(key=lambda x: (-x[0], x[1]))
        return scored_nodes[0][2]
    
    def update_node_load(self, node_id: str, load: float):
        """Update node load information."""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].load = load
                self.nodes[node_id].last_heartbeat = time.time()
    
    def update_node_status(self, node_id: str, status: str):
        """Update node status."""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                logger.debug(f"Node {node_id} status: {status}")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        with self.lock:
            total_nodes = len(self.nodes)
            idle_nodes = sum(1 for n in self.nodes.values() if n.status == "idle")
            busy_nodes = sum(1 for n in self.nodes.values() if n.status == "busy")
            failed_nodes = sum(1 for n in self.nodes.values() if n.status == "failed")
            avg_load = np.mean([n.load for n in self.nodes.values()]) if self.nodes else 0
            
            return {
                "total_nodes": total_nodes,
                "idle_nodes": idle_nodes,
                "busy_nodes": busy_nodes,
                "failed_nodes": failed_nodes,
                "average_load": avg_load,
                "strategies": self.strategy
            }


class TaskScheduler:
    """Scheduler for distributed quantum computing tasks."""
    
    def __init__(self, load_balancer: LoadBalancer, max_concurrent_tasks: int = 100):
        self.load_balancer = load_balancer
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = Queue()
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, str] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.scheduler_thread = None
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start the task scheduler."""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")
    
    def stop(self):
        """Stop the task scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        logger.info("Task scheduler stopped")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for execution."""
        self.task_queue.put(task)
        logger.debug(f"Task submitted: {task.task_id}")
        return task.task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of completed task."""
        start_time = time.time()
        
        while True:
            with self.lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
                
                if task_id in self.failed_tasks:
                    raise RuntimeError(f"Task {task_id} failed: {self.failed_tasks[task_id]}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(0.1)
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Get next task from queue
                try:
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Check dependencies
                if not self._dependencies_satisfied(task):
                    # Put task back in queue
                    self.task_queue.put(task)
                    time.sleep(0.1)
                    continue
                
                # Find available node
                node = self.load_balancer.select_node(task)
                if node is None:
                    # No available nodes, put task back
                    self.task_queue.put(task)
                    time.sleep(0.5)
                    continue
                
                # Execute task
                self._execute_task(task, node)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        with self.lock:
            return all(dep in self.completed_tasks for dep in task.dependencies)
    
    def _execute_task(self, task: DistributedTask, node: ComputeNode):
        """Execute task on selected node."""
        with self.lock:
            self.running_tasks[task.task_id] = task
        
        # Update node status
        self.load_balancer.update_node_status(node.node_id, "busy")
        
        # Submit task to executor
        future = self.executor.submit(self._run_task_on_node, task, node)
        
        # Add completion callback
        future.add_done_callback(
            lambda f: self._task_completed(task, node, f)
        )
        
        logger.debug(f"Task {task.task_id} assigned to node {node.node_id}")
    
    def _run_task_on_node(self, task: DistributedTask, node: ComputeNode) -> Any:
        """Run task on specific node."""
        try:
            # Simulate network communication and task execution
            # In real implementation, this would use RPC or message passing
            
            # Get the function to execute
            func = self._get_function(task.function_name)
            
            # Execute the function
            result = func(*task.args, **task.kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed on node {node.node_id}: {e}")
            raise
    
    def _get_function(self, function_name: str) -> Callable:
        """Get function by name for execution."""
        # This is a simplified implementation
        # In practice, would use a function registry or RPC
        
        if function_name == "matrix_multiply":
            return self._matrix_multiply
        elif function_name == "fft_computation":
            return self._fft_computation
        elif function_name == "tensor_contraction":
            return self._tensor_contraction
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    def _matrix_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Distributed matrix multiplication."""
        return jnp.dot(a, b)
    
    def _fft_computation(self, x: jnp.ndarray, axes: Optional[tuple] = None) -> jnp.ndarray:
        """Distributed FFT computation."""
        if axes:
            return jnp.fft.fftn(x, axes=axes)
        else:
            return jnp.fft.fft(x)
    
    def _tensor_contraction(self, *tensors, equation: str) -> jnp.ndarray:
        """Distributed tensor contraction."""
        return jnp.einsum(equation, *tensors)
    
    def _task_completed(self, task: DistributedTask, node: ComputeNode, future):
        """Handle task completion."""
        with self.lock:
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
        
        # Update node status
        self.load_balancer.update_node_status(node.node_id, "idle")
        
        try:
            result = future.result()
            with self.lock:
                self.completed_tasks[task.task_id] = result
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            with self.lock:
                self.failed_tasks[task.task_id] = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.lock:
            return {
                "queued_tasks": self.task_queue.qsize(),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "max_concurrent": self.max_concurrent_tasks
            }


class DistributedQuantumOperator:
    """Base class for distributed quantum neural operators."""
    
    def __init__(self, scheduler: TaskScheduler):
        self.scheduler = scheduler
        self.task_counter = 0
        self.lock = threading.Lock()
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        with self.lock:
            self.task_counter += 1
            return f"task_{self.task_counter}_{int(time.time())}"
    
    def distributed_matrix_multiply(self, a: jnp.ndarray, b: jnp.ndarray,
                                   block_size: int = 1024) -> jnp.ndarray:
        """
        Distributed matrix multiplication with block decomposition.
        
        Args:
            a: First matrix
            b: Second matrix
            block_size: Size of blocks for distribution
            
        Returns:
            Product matrix
        """
        m, k = a.shape
        k2, n = b.shape
        
        if k != k2:
            raise ValueError(f"Matrix dimensions incompatible: {a.shape} x {b.shape}")
        
        # If matrices are small, compute locally
        if m * n * k < block_size ** 3:
            return jnp.dot(a, b)
        
        # Partition matrices into blocks
        block_tasks = []
        results_grid = {}
        
        for i in range(0, m, block_size):
            for j in range(0, n, block_size):
                i_end = min(i + block_size, m)
                j_end = min(j + block_size, n)
                
                # Compute block (i,j) as sum of products
                block_result_tasks = []
                
                for l in range(0, k, block_size):
                    l_end = min(l + block_size, k)
                    
                    a_block = a[i:i_end, l:l_end]
                    b_block = b[l:l_end, j:j_end]
                    
                    task_id = self._generate_task_id()
                    task = DistributedTask(
                        task_id=task_id,
                        function_name="matrix_multiply",
                        args=(a_block, b_block),
                        kwargs={},
                        node_requirements=["cpu", "numpy"]
                    )
                    
                    self.scheduler.submit_task(task)
                    block_result_tasks.append(task_id)
                
                results_grid[(i//block_size, j//block_size)] = block_result_tasks
        
        # Collect results and assemble final matrix
        result = jnp.zeros((m, n))
        
        for (i_block, j_block), task_ids in results_grid.items():
            i_start = i_block * block_size
            j_start = j_block * block_size
            i_end = min(i_start + block_size, m)
            j_end = min(j_start + block_size, n)
            
            # Sum partial results for this block
            block_sum = None
            for task_id in task_ids:
                partial_result = self.scheduler.get_task_result(task_id, timeout=30.0)
                if block_sum is None:
                    block_sum = partial_result
                else:
                    block_sum += partial_result
            
            result = result.at[i_start:i_end, j_start:j_end].set(block_sum)
        
        return result
    
    def distributed_fft(self, x: jnp.ndarray, axes: Optional[tuple] = None) -> jnp.ndarray:
        """
        Distributed FFT computation.
        
        Args:
            x: Input tensor
            axes: Axes along which to compute FFT
            
        Returns:
            FFT result
        """
        # For large tensors, partition along first dimension
        if x.size > 1024 * 1024:  # 1M elements
            batch_size = x.shape[0]
            chunk_size = max(1, batch_size // 4)  # Use 4 chunks
            
            if batch_size > chunk_size:
                tasks = []
                
                for i in range(0, batch_size, chunk_size):
                    end_i = min(i + chunk_size, batch_size)
                    chunk = x[i:end_i]
                    
                    task_id = self._generate_task_id()
                    task = DistributedTask(
                        task_id=task_id,
                        function_name="fft_computation",
                        args=(chunk,),
                        kwargs={"axes": axes},
                        node_requirements=["gpu", "fft"]
                    )
                    
                    self.scheduler.submit_task(task)
                    tasks.append(task_id)
                
                # Collect results
                results = []
                for task_id in tasks:
                    result = self.scheduler.get_task_result(task_id, timeout=60.0)
                    results.append(result)
                
                return jnp.concatenate(results, axis=0)
        
        # Compute locally for small tensors
        if axes:
            return jnp.fft.fftn(x, axes=axes)
        else:
            return jnp.fft.fft(x)
    
    def distributed_einsum(self, equation: str, *operands) -> jnp.ndarray:
        """
        Distributed Einstein summation.
        
        Args:
            equation: Einstein summation equation
            *operands: Input tensors
            
        Returns:
            Contracted result
        """
        # Estimate computation complexity
        total_elements = 1
        for operand in operands:
            total_elements *= operand.size
        
        # Use distributed computation for large operations
        if total_elements > 10**8:  # 100M elements
            task_id = self._generate_task_id()
            task = DistributedTask(
                task_id=task_id,
                function_name="tensor_contraction",
                args=operands,
                kwargs={"equation": equation},
                node_requirements=["gpu", "tensor"]
            )
            
            self.scheduler.submit_task(task)
            return self.scheduler.get_task_result(task_id, timeout=300.0)
        
        # Compute locally
        return jnp.einsum(equation, *operands)


class DistributedTrainingCoordinator:
    """Coordinates distributed training across multiple nodes."""
    
    def __init__(self, scheduler: TaskScheduler, synchronization_method: str = "allreduce"):
        self.scheduler = scheduler
        self.synchronization_method = synchronization_method
        self.training_state = {}
        self.gradient_accumulator = {}
        self.lock = threading.Lock()
    
    def distribute_training_batch(self, batch_data: Dict[str, jnp.ndarray],
                                 model_params: Dict[str, jnp.ndarray],
                                 n_workers: int = 4) -> Dict[str, jnp.ndarray]:
        """
        Distribute training batch across multiple workers.
        
        Args:
            batch_data: Training batch
            model_params: Current model parameters
            n_workers: Number of worker nodes
            
        Returns:
            Aggregated gradients
        """
        batch_size = batch_data['inputs'].shape[0]
        chunk_size = max(1, batch_size // n_workers)
        
        # Submit training tasks to workers
        tasks = []
        for i in range(n_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)
            
            if start_idx >= batch_size:
                break
            
            # Create data chunk
            chunk_data = {
                key: value[start_idx:end_idx] 
                for key, value in batch_data.items()
            }
            
            task_id = f"training_worker_{i}_{int(time.time())}"
            task = DistributedTask(
                task_id=task_id,
                function_name="compute_gradients",
                args=(chunk_data, model_params),
                kwargs={},
                node_requirements=["gpu", "training"]
            )
            
            self.scheduler.submit_task(task)
            tasks.append(task_id)
        
        # Collect gradients from workers
        worker_gradients = []
        for task_id in tasks:
            gradients = self.scheduler.get_task_result(task_id, timeout=120.0)
            worker_gradients.append(gradients)
        
        # Aggregate gradients
        return self._aggregate_gradients(worker_gradients)
    
    def _aggregate_gradients(self, worker_gradients: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Aggregate gradients from multiple workers."""
        if not worker_gradients:
            return {}
        
        # Initialize aggregated gradients
        aggregated = {}
        for key in worker_gradients[0].keys():
            aggregated[key] = jnp.zeros_like(worker_gradients[0][key])
        
        # Sum gradients
        for gradients in worker_gradients:
            for key, grad in gradients.items():
                aggregated[key] += grad
        
        # Average gradients
        n_workers = len(worker_gradients)
        for key in aggregated.keys():
            aggregated[key] /= n_workers
        
        return aggregated
    
    def synchronize_parameters(self, local_params: Dict[str, jnp.ndarray],
                              worker_nodes: List[str]) -> Dict[str, jnp.ndarray]:
        """Synchronize parameters across worker nodes."""
        if self.synchronization_method == "allreduce":
            return self._allreduce_synchronization(local_params, worker_nodes)
        elif self.synchronization_method == "parameter_server":
            return self._parameter_server_synchronization(local_params, worker_nodes)
        else:
            raise ValueError(f"Unknown synchronization method: {self.synchronization_method}")
    
    def _allreduce_synchronization(self, local_params: Dict[str, jnp.ndarray],
                                  worker_nodes: List[str]) -> Dict[str, jnp.ndarray]:
        """All-reduce parameter synchronization."""
        # Simplified all-reduce implementation
        # In practice, would use optimized communication libraries
        
        synchronized_params = {}
        for key, param in local_params.items():
            # Average parameters across all workers
            synchronized_params[key] = param  # Simplified
        
        return synchronized_params
    
    def _parameter_server_synchronization(self, local_params: Dict[str, jnp.ndarray],
                                        worker_nodes: List[str]) -> Dict[str, jnp.ndarray]:
        """Parameter server synchronization."""
        # Store parameters on parameter server
        with self.lock:
            self.training_state["global_params"] = local_params
        
        return local_params


def create_distributed_cluster(node_configs: List[Dict[str, Any]]) -> Tuple[LoadBalancer, TaskScheduler]:
    """
    Create a distributed quantum computing cluster.
    
    Args:
        node_configs: List of node configuration dictionaries
        
    Returns:
        Configured load balancer and task scheduler
    """
    # Create load balancer
    load_balancer = LoadBalancer(strategy="capability_based")
    
    # Register compute nodes
    for i, config in enumerate(node_configs):
        node = ComputeNode(
            node_id=f"node_{i}",
            host=config.get("host", "localhost"),
            port=config.get("port", 8000 + i),
            capabilities=config.get("capabilities", ["cpu", "numpy"])
        )
        load_balancer.register_node(node)
    
    # Create task scheduler
    scheduler = TaskScheduler(load_balancer, max_concurrent_tasks=50)
    scheduler.start()
    
    logger.info(f"Created distributed cluster with {len(node_configs)} nodes")
    
    return load_balancer, scheduler