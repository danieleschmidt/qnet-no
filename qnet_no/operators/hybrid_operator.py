"""Hybrid classical-quantum neural operator implementation."""

from typing import Dict, Any, Optional, Tuple, Union
import jax
import jax.numpy as jnp  
import flax.linen as nn
from flax.training import train_state
import optax
from .quantum_fno import QuantumFourierNeuralOperator
from .quantum_deeponet import QuantumDeepONet
from ..networks.photonic_network import PhotonicNetwork
from ..utils.classical_layers import ResidualBlock, AttentionLayer


class HybridNeuralOperator(nn.Module):
    """
    Hybrid classical-quantum neural operator combining multiple architectures.
    
    Intelligently routes computation between classical and quantum processors
    based on problem characteristics and available quantum resources.
    """
    
    fno_modes: int = 16
    deeponet_trunk_dim: int = 64
    classical_width: int = 128
    schmidt_rank: int = 8
    n_layers: int = 4
    fusion_strategy: str = "adaptive"  # "weighted", "gated", "adaptive"
    
    def setup(self):
        # Quantum components
        self.quantum_fno = QuantumFourierNeuralOperator(
            modes=self.fno_modes,
            width=self.classical_width,
            schmidt_rank=self.schmidt_rank,
            n_layers=self.n_layers
        )
        
        self.quantum_deeponet = QuantumDeepONet(
            trunk_dim=self.deeponet_trunk_dim,
            n_layers=self.n_layers,
            schmidt_rank=self.schmidt_rank
        )
        
        # Classical components  
        self.classical_backbone = [
            ResidualBlock(self.classical_width) for _ in range(self.n_layers)
        ]
        
        self.attention_layer = AttentionLayer(
            embed_dim=self.classical_width,
            num_heads=8
        )
        
        # Fusion layers
        if self.fusion_strategy == "weighted":
            self.fusion_weights = self.param(
                'fusion_weights', nn.initializers.uniform(), (3,)
            )
        elif self.fusion_strategy == "gated":
            self.gate_network = nn.Dense(3)
        elif self.fusion_strategy == "adaptive":
            self.complexity_estimator = nn.Dense(1)
            self.adaptive_router = nn.Dense(3)
        
        self.output_projection = nn.Dense(1)
    
    def __call__(self, inputs: Dict[str, jnp.ndarray], 
                 network: PhotonicNetwork) -> jnp.ndarray:
        """
        Forward pass through hybrid operator.
        
        Args:
            inputs: Dictionary containing different input formats:
                   - 'grid_data': For FNO processing [batch, *spatial, channels]  
                   - 'function_data': For DeepONet branch [batch, n_sensors]
                   - 'query_points': For DeepONet trunk [batch, n_queries, dim]
            network: Quantum photonic network
            
        Returns:
            Operator output [batch, n_queries] or [batch, *spatial, 1]
        """
        outputs = []
        
        # Classical processing path
        if 'grid_data' in inputs:
            x_classical = inputs['grid_data']
            
            # Classical residual backbone
            for residual_block in self.classical_backbone:
                x_classical = residual_block(x_classical)
            
            # Self-attention for long-range dependencies
            batch_size = x_classical.shape[0]
            spatial_shape = x_classical.shape[1:-1]
            n_features = x_classical.shape[-1]
            
            # Flatten spatial dimensions for attention
            x_flat = x_classical.reshape(batch_size, -1, n_features)
            x_attended = self.attention_layer(x_flat)
            x_classical = x_attended.reshape(batch_size, *spatial_shape, n_features)
            
            outputs.append(x_classical)
        
        # Quantum FNO path
        if 'grid_data' in inputs:
            fno_output = self.quantum_fno(inputs['grid_data'], network)
            outputs.append(fno_output)
        
        # Quantum DeepONet path  
        if 'function_data' in inputs and 'query_points' in inputs:
            deeponet_output = self.quantum_deeponet(
                inputs['function_data'], 
                inputs['query_points'], 
                network
            )
            # Expand dimensions to match other outputs if needed
            if len(deeponet_output.shape) == 2:
                deeponet_output = deeponet_output[..., None]
            outputs.append(deeponet_output)
        
        # Fusion strategy
        if len(outputs) == 1:
            fused_output = outputs[0]
        else:
            fused_output = self._fuse_outputs(outputs, inputs, network)
        
        # Final output projection
        return self.output_projection(fused_output)
    
    def _fuse_outputs(self, outputs: list, inputs: Dict[str, jnp.ndarray],
                     network: PhotonicNetwork) -> jnp.ndarray:
        """Fuse multiple operator outputs using specified strategy."""
        
        if self.fusion_strategy == "weighted":
            # Simple weighted combination
            weights = nn.softmax(self.fusion_weights[:len(outputs)])
            fused = sum(w * out for w, out in zip(weights, outputs))
            
        elif self.fusion_strategy == "gated":
            # Learned gating based on input characteristics
            if 'grid_data' in inputs:
                gate_input = jnp.mean(inputs['grid_data'], axis=(1, 2), keepdims=True)
            else:
                gate_input = jnp.mean(inputs['function_data'], axis=1, keepdims=True)
            
            gates = nn.softmax(self.gate_network(gate_input))
            gates = gates[:, :len(outputs)]
            
            fused = sum(g[..., i:i+1] * out for i, (g, out) in enumerate(zip(gates, outputs)))
            
        elif self.fusion_strategy == "adaptive":
            # Adaptive routing based on problem complexity
            complexity_features = self._extract_complexity_features(inputs)
            complexity_score = nn.sigmoid(self.complexity_estimator(complexity_features))
            
            routing_weights = nn.softmax(self.adaptive_router(complexity_features))
            routing_weights = routing_weights[:, :len(outputs)]
            
            # Weight quantum components more for complex problems
            quantum_boost = complexity_score * 0.5
            if len(outputs) >= 2:  # Has quantum components
                routing_weights = routing_weights.at[:, 1:].multiply(1 + quantum_boost)
                routing_weights = routing_weights / jnp.sum(routing_weights, axis=1, keepdims=True)
            
            fused = sum(w[..., i:i+1] * out for i, (w, out) in enumerate(zip(routing_weights, outputs)))
        
        return fused
    
    def _extract_complexity_features(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Extract features indicating problem complexity for adaptive routing."""
        features = []
        
        if 'grid_data' in inputs:
            x = inputs['grid_data']
            # Spatial variation
            grad_norm = jnp.mean(jnp.abs(jnp.gradient(x, axis=1)), axis=(1, 2))
            features.append(grad_norm)
            
            # Frequency content (high freq indicates complexity)
            x_fft = jnp.fft.fft2(x[..., 0])
            high_freq_content = jnp.mean(jnp.abs(x_fft[:, x_fft.shape[1]//4:, x_fft.shape[2]//4:]), axis=(1, 2))
            features.append(high_freq_content[..., None])
        
        if 'function_data' in inputs:
            u = inputs['function_data']
            # Function variation
            func_variation = jnp.std(u, axis=1, keepdims=True)
            features.append(func_variation)
        
        return jnp.concatenate(features, axis=-1) if features else jnp.ones((inputs[list(inputs.keys())[0]].shape[0], 1))
    
    def create_train_state(self, rng: jax.random.PRNGKey, 
                          input_shapes: Dict[str, Tuple[int, ...]],
                          network: PhotonicNetwork, learning_rate: float = 1e-3) -> train_state.TrainState:
        """Create training state with optimizer."""
        dummy_inputs = {k: jnp.ones(shape) for k, shape in input_shapes.items()}
        params = self.init(rng, dummy_inputs, network)['params']
        
        tx = optax.adamw(learning_rate, weight_decay=1e-4)
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=tx
        )
    
    def fit(self, train_data: Dict[str, jnp.ndarray], network: PhotonicNetwork,
            epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> Dict[str, Any]:
        """Train the hybrid neural operator."""
        rng = jax.random.PRNGKey(42)
        
        # Determine input shapes from training data
        input_shapes = {}
        for key in ['grid_data', 'function_data', 'query_points']:
            if key in train_data:
                input_shapes[key] = (batch_size, *train_data[key].shape[1:])
        
        state = self.create_train_state(rng, input_shapes, network, lr)
        
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
        n_samples = len(train_data['targets'])
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            perm = jax.random.permutation(rng, n_samples)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = perm[start_idx:end_idx]
                
                batch_inputs = {}
                for key in input_shapes.keys():
                    batch_inputs[key] = train_data[key][batch_indices]
                
                batch_targets = train_data['targets'][batch_indices]
                
                state, loss = train_step(state, batch_inputs, batch_targets)
                epoch_losses.append(loss)
            
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.trained_params = state.params
        return {'losses': losses, 'final_state': state}
    
    def predict(self, test_data: Dict[str, jnp.ndarray], 
                network: PhotonicNetwork) -> jnp.ndarray:
        """Make predictions using trained model."""
        if not hasattr(self, 'trained_params'):
            raise ValueError("Model must be trained before making predictions")
        
        # Remove targets from test_data for prediction
        input_data = {k: v for k, v in test_data.items() if k != 'targets'}
        predictions = self.apply({'params': self.trained_params}, input_data, network)
        return predictions