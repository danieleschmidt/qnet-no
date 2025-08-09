"""Quantum Fourier Neural Operator implementation."""

from typing import Dict, Any, Optional, Tuple
import time
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import logging
from ..networks.photonic_network import PhotonicNetwork
from ..utils.quantum_fourier import quantum_fourier_modes
from ..utils.tensor_ops import tensor_product_einsum
from ..utils.validation import (
    validate_tensor_shape, validate_operator_parameters, 
    validate_training_parameters, log_validation_result
)
from ..utils.error_handling import (
    error_boundary, OperatorError, TrainingError, ErrorSeverity, 
    monitor_resources, safe_quantum_operation
)
from ..utils.performance import (
    MemoryPool, ComputationCache, PerformanceProfiler, 
    AdaptiveBatchSize
)
from ..utils.distributed import (
    DistributedQuantumOperator, TaskScheduler, LoadBalancer, 
    create_distributed_cluster
)
from ..utils.metrics import (
    get_metrics_collector, record_quantum_operation, record_training_step
)

logger = logging.getLogger(__name__)


class QuantumSpectralConv(nn.Module):
    """Quantum spectral convolution layer using photonic QPUs."""
    
    modes: int
    schmidt_rank: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, network: PhotonicNetwork) -> jnp.ndarray:
        batch_size, *spatial_dims, channels = x.shape
        
        # Quantum Fourier transform on distributed QPUs
        x_ft = quantum_fourier_modes(x, self.modes, network, self.schmidt_rank)
        
        # Learnable quantum spectral weights
        weights_real = self.param(
            'weights_real',
            nn.initializers.glorot_uniform(),
            (self.modes, self.modes, channels, channels, self.schmidt_rank)
        )
        weights_imag = self.param(
            'weights_imag', 
            nn.initializers.glorot_uniform(),
            (self.modes, self.modes, channels, channels, self.schmidt_rank)
        )
        
        weights = weights_real + 1j * weights_imag
        
        # Quantum tensor contraction across Schmidt ranks
        out_ft = tensor_product_einsum(
            'b...nc,mncos->b...mos', 
            x_ft, 
            weights,
            network
        )
        
        # Inverse quantum Fourier transform  
        out = quantum_fourier_modes(out_ft, self.modes, network, self.schmidt_rank, inverse=True)
        return out.real


class QuantumFourierNeuralOperator(nn.Module, DistributedQuantumOperator):
    """
    Quantum-enhanced Fourier Neural Operator for distributed quantum computing.
    
    Implements neural operators using quantum photonic processing units (QPUs)
    connected via entanglement channels for quantum-accelerated PDE solving.
    Enhanced with performance optimization and distributed scaling capabilities.
    """
    
    modes: int = 16
    width: int = 64
    schmidt_rank: int = 8
    n_layers: int = 4
    
    def setup(self):
        self.fc_in = nn.Dense(self.width)
        
        self.conv_layers = [
            QuantumSpectralConv(self.modes, self.schmidt_rank)
            for _ in range(self.n_layers)
        ]
        
        self.w_layers = [
            nn.Dense(self.width) for _ in range(self.n_layers)
        ]
        
        self.fc_out1 = nn.Dense(128)
        self.fc_out2 = nn.Dense(1)
        
        # Initialize performance optimization components
        self.memory_pool = MemoryPool(max_pool_size=500 * 1024 * 1024, cleanup_threshold=0.8)
        self.computation_cache = ComputationCache(
            max_memory_gb=2.0, 
            cache_dir="./cache/qfno",
            compression=True
        )
        self.profiler = PerformanceProfiler()
        self.batch_sizer = AdaptiveBatchSize(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=256,
            target_memory_gb=4.0
        )
        
        # Distributed computing components (initialized when needed)
        self.scheduler = None
        self.distributed_enabled = False
        
        # Metrics collection
        self.metrics_collector = get_metrics_collector()
    
    @record_quantum_operation
    def __call__(self, x: jnp.ndarray, network: PhotonicNetwork, 
                 use_cache: bool = True, use_distributed: bool = False) -> jnp.ndarray:
        """Forward pass with performance optimization and optional distributed computing."""
        
        # Record quantum metrics
        self.metrics_collector.record_quantum_metrics(
            schmidt_rank=self.schmidt_rank,
            circuit_depth=self.n_layers
        )
        
        # Start profiling
        with self.profiler.profile_context("qfno_forward"):
            # Check cache for computed results
            cache_key = None
            if use_cache:
                cache_key = self.computation_cache.generate_cache_key(
                    "qfno_forward", x.shape, self.modes, self.schmidt_rank
                )
                cached_result = self.computation_cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for forward pass: {cache_key}")
                    return cached_result
            
            # Get tensors from memory pool for intermediate computations
            x_tensor = self.memory_pool.get_tensor(x.shape, x.dtype)
            x_tensor = x_tensor.at[:].set(x)
            
            try:
                # Input layer
                x_tensor = self.fc_in(x_tensor)
                
                # Main layers with optional distributed processing
                for i, (conv_layer, w_layer) in enumerate(zip(self.conv_layers, self.w_layers)):
                    if use_distributed and self.distributed_enabled and x_tensor.size > 1024*1024:
                        # Use distributed computation for large tensors
                        x1 = self._distributed_conv_layer(x_tensor, conv_layer, network, i)
                    else:
                        x1 = conv_layer(x_tensor, network)
                    
                    x2 = w_layer(x_tensor)
                    x_tensor = nn.gelu(x1 + x2)
                
                # Output layers
                x_tensor = nn.gelu(self.fc_out1(x_tensor))
                result = self.fc_out2(x_tensor)
                
                # Cache result if enabled
                if use_cache and cache_key:
                    self.computation_cache.put(cache_key, result)
                
                return result
                
            finally:
                # Return tensor to memory pool
                self.memory_pool.return_tensor(x_tensor)
    
    def create_train_state(self, rng: jax.random.PRNGKey, input_shape: Tuple[int, ...], 
                          network: PhotonicNetwork, learning_rate: float = 1e-3) -> train_state.TrainState:
        """Create training state with optimizer."""
        dummy_input = jnp.ones(input_shape)
        params = self.init(rng, dummy_input, network)['params']
        
        tx = optax.adam(learning_rate)  
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=tx
        )
    
    @error_boundary(TrainingError, ErrorSeverity.HIGH)
    @monitor_resources("qfno_training")
    @record_training_step
    def fit(self, train_data: Dict[str, jnp.ndarray], network: PhotonicNetwork,
            epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> Dict[str, Any]:
        """Train the quantum neural operator with comprehensive validation and error handling."""
        logger.info(f"Starting QFNO training: epochs={epochs}, lr={lr}, batch_size={batch_size}")
        
        # Validate training parameters
        validation_result = validate_training_parameters(epochs, batch_size, lr, len(train_data['inputs']))
        log_validation_result(validation_result, "training parameters")
        
        if not validation_result.is_valid:
            raise TrainingError(f"Invalid training parameters: {validation_result.errors}")
        
        # Validate operator parameters
        op_validation = validate_operator_parameters(
            modes=self.modes, width=self.width, 
            schmidt_rank=self.schmidt_rank, n_layers=self.n_layers
        )
        log_validation_result(op_validation, "operator parameters")
        
        # Validate input data
        input_validation = validate_tensor_shape(
            train_data['inputs'], 
            min_dims=3, max_dims=5, 
            name="training inputs"
        )
        target_validation = validate_tensor_shape(
            train_data['targets'],
            min_dims=3, max_dims=5,
            name="training targets"
        )
        
        if not input_validation.is_valid:
            raise TrainingError(f"Invalid input data: {input_validation.errors}")
        if not target_validation.is_valid:
            raise TrainingError(f"Invalid target data: {target_validation.errors}")
        
        # Check data consistency
        if train_data['inputs'].shape[0] != train_data['targets'].shape[0]:
            raise TrainingError(f"Input batch size {train_data['inputs'].shape[0]} != target batch size {train_data['targets'].shape[0]}")
        
        rng = jax.random.PRNGKey(42)
        
        # Initialize training state with error handling
        try:
            input_shape = (batch_size, *train_data['inputs'].shape[1:])
            state = self.create_train_state(rng, input_shape, network, lr)
            logger.info("Training state initialized successfully")
        except Exception as e:
            raise TrainingError(f"Failed to initialize training state: {e}", severity=ErrorSeverity.HIGH)
        
        @jax.jit
        def train_step(state, batch_inputs, batch_targets):
            def loss_fn(params):
                predictions = self.apply({'params': params}, batch_inputs, network)
                return jnp.mean((predictions - batch_targets) ** 2)
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        # Training loop
        losses = []
        n_batches = len(train_data['inputs']) // batch_size
        epoch_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            perm = jax.random.permutation(rng, len(train_data['inputs']))
            inputs_shuffled = train_data['inputs'][perm]
            targets_shuffled = train_data['targets'][perm]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_inputs = inputs_shuffled[start_idx:end_idx]
                batch_targets = targets_shuffled[start_idx:end_idx]
                
                state, loss = train_step(state, batch_inputs, batch_targets)
                epoch_losses.append(loss)
            
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            losses.append(avg_loss)
            
            # Record training metrics
            self.metrics_collector.record_training_metrics(
                epoch=epoch,
                loss=float(avg_loss),
                learning_rate=lr,
                batch_size=batch_size,
                throughput=len(train_data['inputs']) / (time.time() - epoch_start_time) if 'epoch_start_time' in locals() else 0
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
                
            # Start timing for next epoch
            epoch_start_time = time.time()
        
        self.trained_params = state.params
        
        # Log performance metrics
        training_stats = self.profiler.get_statistics()
        logger.info(f"Training completed - Performance stats: {training_stats}")
        
        return {
            'losses': losses, 
            'final_state': state,
            'performance_stats': training_stats,
            'cache_stats': self.computation_cache.get_statistics(),
            'memory_stats': self.memory_pool.get_statistics()
        }
    
    @error_boundary(OperatorError, ErrorSeverity.MEDIUM)
    @monitor_resources("qfno_prediction") 
    def predict(self, test_data: Dict[str, jnp.ndarray], network: PhotonicNetwork) -> jnp.ndarray:
        """Make predictions using trained model with validation."""
        logger.info("Starting QFNO prediction")
        
        # Check if model is trained
        if not hasattr(self, 'trained_params'):
            raise OperatorError("Model must be trained before making predictions", severity=ErrorSeverity.HIGH)
        
        # Validate input data
        if 'inputs' not in test_data:
            raise OperatorError("Test data must contain 'inputs' key", severity=ErrorSeverity.HIGH)
        
        input_validation = validate_tensor_shape(
            test_data['inputs'],
            min_dims=3, max_dims=5,
            name="prediction inputs"
        )
        
        if not input_validation.is_valid:
            raise OperatorError(f"Invalid prediction inputs: {input_validation.errors}")
        
        log_validation_result(input_validation, "prediction inputs")
        
        # Check network connectivity
        if len(network.quantum_nodes) == 0:
            raise OperatorError("Network has no quantum nodes", severity=ErrorSeverity.HIGH)
        
        try:
            # Make predictions with quantum network
            predictions = self.apply({'params': self.trained_params}, test_data['inputs'], network)
            
            # Validate prediction outputs
            pred_validation = validate_tensor_shape(predictions, name="predictions")
            if not pred_validation.is_valid:
                logger.warning(f"Prediction output validation issues: {pred_validation.warnings}")
            
            logger.info(f"Prediction completed successfully, output shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            raise OperatorError(f"Prediction failed: {e}", severity=ErrorSeverity.HIGH)
    
    def enable_distributed_computing(self, node_configs: list, strategy: str = "capability_based") -> None:
        """Enable distributed computing across multiple nodes."""
        logger.info(f"Enabling distributed computing with {len(node_configs)} nodes")
        
        # Initialize distributed cluster
        load_balancer, scheduler = create_distributed_cluster(node_configs)
        self.scheduler = scheduler
        self.distributed_enabled = True
        
        # Initialize distributed quantum operator
        DistributedQuantumOperator.__init__(self, scheduler)
        
        logger.info("Distributed computing enabled successfully")
    
    def _distributed_conv_layer(self, x: jnp.ndarray, conv_layer, network: PhotonicNetwork, layer_idx: int) -> jnp.ndarray:
        """Execute convolution layer using distributed computation."""
        if not self.distributed_enabled:
            return conv_layer(x, network)
        
        # Split tensor for distributed processing
        batch_size = x.shape[0]
        if batch_size >= 4:  # Only distribute if batch is large enough
            chunk_size = max(1, batch_size // 4)
            chunks = []
            task_ids = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk = x[i:end_idx]
                
                # Submit distributed task
                task_id = self._generate_task_id()
                task = {
                    'task_id': task_id,
                    'function_name': 'quantum_conv_layer',
                    'args': (chunk, conv_layer, network),
                    'kwargs': {'layer_idx': layer_idx},
                    'node_requirements': ['gpu', 'quantum']
                }
                
                self.scheduler.submit_task(task)
                task_ids.append(task_id)
            
            # Collect results
            results = []
            for task_id in task_ids:
                result = self.scheduler.get_task_result(task_id, timeout=60.0)
                results.append(result)
            
            return jnp.concatenate(results, axis=0)
        else:
            return conv_layer(x, network)
    
    def auto_scale_batch_size(self, current_loss: float, memory_usage: float, 
                             throughput: float) -> int:
        """Automatically adjust batch size based on performance metrics."""
        return self.batch_sizer.adjust_batch_size(
            current_loss=current_loss,
            memory_usage_gb=memory_usage / (1024**3),
            throughput=throughput
        )
    
    def enable_auto_scaling(self, network: PhotonicNetwork, target_utilization: float = 0.75) -> None:
        """Enable automatic scaling based on network load and performance."""
        logger.info("Enabling auto-scaling based on network performance")
        
        def scaling_monitor():
            while True:
                # Monitor network performance
                network_stats = network.get_network_statistics()
                avg_load = network_stats.get('average_node_load', 0.0)
                
                # Auto-scale based on load
                if avg_load > target_utilization and not self.distributed_enabled:
                    logger.info(f"High load detected ({avg_load:.2f}), enabling distributed computing")
                    # Auto-create additional compute nodes
                    additional_nodes = self._create_additional_nodes(network_stats)
                    if additional_nodes:
                        self.enable_distributed_computing(additional_nodes)
                
                # Adjust memory pool size based on usage
                memory_stats = self.memory_pool.get_statistics()
                if memory_stats['utilization'] > 0.9:
                    logger.info("High memory utilization, expanding memory pool")
                    self.memory_pool.expand_pool(factor=1.5)
                
                # Clean up cache if getting too large
                cache_stats = self.computation_cache.get_statistics()
                if cache_stats['memory_usage_gb'] > 1.5:
                    logger.info("Cache size limit reached, performing cleanup")
                    self.computation_cache.cleanup_cache(target_size_gb=1.0)
                
                # Sleep before next check
                import time
                time.sleep(30)  # Check every 30 seconds
        
        # Start monitoring thread
        import threading
        monitor_thread = threading.Thread(target=scaling_monitor, daemon=True)
        monitor_thread.start()
        
        logger.info("Auto-scaling enabled")
    
    def _create_additional_nodes(self, network_stats: dict) -> list:
        """Create additional compute nodes based on current network state."""
        current_node_count = network_stats.get('total_nodes', 1)
        
        # Create 2-4 additional nodes based on load
        additional_node_configs = []
        for i in range(2, min(5, current_node_count + 3)):
            config = {
                'host': 'localhost',
                'port': 8000 + current_node_count + i,
                'capabilities': ['gpu', 'quantum', 'fft', 'tensor']
            }
            additional_node_configs.append(config)
        
        return additional_node_configs
    
    def get_scaling_statistics(self) -> dict:
        """Get comprehensive scaling and performance statistics."""
        stats = {
            'distributed_enabled': self.distributed_enabled,
            'performance_stats': self.profiler.get_statistics(),
            'memory_stats': self.memory_pool.get_statistics(),
            'cache_stats': self.computation_cache.get_statistics(),
            'batch_sizer_stats': self.batch_sizer.get_statistics()
        }
        
        if self.distributed_enabled and self.scheduler:
            stats['scheduler_stats'] = self.scheduler.get_scheduler_stats()
        
        return stats
    
    def cleanup_resources(self) -> None:
        """Clean up all scaling resources."""
        logger.info("Cleaning up scaling resources")
        
        # Cleanup memory pool
        self.memory_pool.cleanup()
        
        # Cleanup computation cache
        self.computation_cache.cleanup_cache()
        
        # Stop distributed scheduler if running
        if self.distributed_enabled and self.scheduler:
            self.scheduler.stop()
        
        logger.info("Resource cleanup completed")