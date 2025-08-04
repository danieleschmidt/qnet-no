"""Quantum DeepONet implementation for distributed operator learning."""

from typing import Dict, Any, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from ..networks.photonic_network import PhotonicNetwork
from ..utils.quantum_encoding import quantum_feature_map
from ..utils.tensor_ops import distributed_dot_product


class QuantumBranchNet(nn.Module):
    """Quantum branch network for encoding input functions."""
    
    trunk_dim: int = 64
    n_layers: int = 4
    schmidt_rank: int = 8
    
    @nn.compact
    def __call__(self, u: jnp.ndarray, network: PhotonicNetwork) -> jnp.ndarray:
        """
        Process input function u through quantum encoding.
        
        Args:
            u: Input function values [batch, n_sensors]
            network: Quantum photonic network
            
        Returns:
            Encoded function features [batch, trunk_dim]
        """
        # Quantum feature encoding on distributed QPUs
        x = quantum_feature_map(u, network, self.schmidt_rank)
        
        # Classical neural network layers
        for i in range(self.n_layers):
            x = nn.Dense(self.trunk_dim)(x)
            x = nn.gelu(x)
        
        # Final quantum encoding layer
        x = nn.Dense(self.trunk_dim)(x) 
        return x


class QuantumTrunkNet(nn.Module):
    """Quantum trunk network for encoding query locations."""
    
    trunk_dim: int = 64
    n_layers: int = 4
    schmidt_rank: int = 8
    
    @nn.compact  
    def __call__(self, y: jnp.ndarray, network: PhotonicNetwork) -> jnp.ndarray:
        """
        Process query locations through quantum encoding.
        
        Args:
            y: Query locations [batch, n_queries, spatial_dim]
            network: Quantum photonic network
            
        Returns:
            Encoded location features [batch, n_queries, trunk_dim]
        """
        batch_size, n_queries, spatial_dim = y.shape
        
        # Reshape for processing
        y_flat = y.reshape(-1, spatial_dim)
        
        # Quantum spatial encoding
        x = quantum_feature_map(y_flat, network, self.schmidt_rank)
        
        # Classical trunk network
        for i in range(self.n_layers):
            x = nn.Dense(self.trunk_dim)(x)
            x = nn.gelu(x)
        
        x = nn.Dense(self.trunk_dim)(x)
        
        # Reshape back
        x = x.reshape(batch_size, n_queries, self.trunk_dim)
        return x


class QuantumDeepONet(nn.Module):
    """
    Quantum-enhanced DeepONet for distributed operator learning.
    
    Implements Deep Operator Networks using quantum photonic processing
    for learning mappings between function spaces on distributed QPUs.
    """
    
    trunk_dim: int = 64
    n_layers: int = 4
    schmidt_rank: int = 8
    
    def setup(self):
        self.branch_net = QuantumBranchNet(
            trunk_dim=self.trunk_dim,
            n_layers=self.n_layers, 
            schmidt_rank=self.schmidt_rank
        )
        
        self.trunk_net = QuantumTrunkNet(
            trunk_dim=self.trunk_dim,
            n_layers=self.n_layers,
            schmidt_rank=self.schmidt_rank  
        )
        
        self.bias = self.param('bias', nn.initializers.zeros, ())
    
    def __call__(self, u: jnp.ndarray, y: jnp.ndarray, 
                 network: PhotonicNetwork) -> jnp.ndarray:
        """
        Evaluate operator at query points.
        
        Args:
            u: Input functions [batch, n_sensors]
            y: Query locations [batch, n_queries, spatial_dim] 
            network: Quantum photonic network
            
        Returns:
            Operator output [batch, n_queries]
        """
        # Encode input function and query locations
        branch_out = self.branch_net(u, network)  # [batch, trunk_dim]
        trunk_out = self.trunk_net(y, network)    # [batch, n_queries, trunk_dim]
        
        # Quantum-distributed dot product across entangled nodes
        outputs = distributed_dot_product(
            branch_out[:, None, :],  # [batch, 1, trunk_dim]
            trunk_out,               # [batch, n_queries, trunk_dim]
            network
        )  # [batch, n_queries]
        
        return outputs + self.bias
    
    def create_train_state(self, rng: jax.random.PRNGKey, 
                          u_shape: Tuple[int, ...], y_shape: Tuple[int, ...],
                          network: PhotonicNetwork, learning_rate: float = 1e-3) -> train_state.TrainState:
        """Create training state with optimizer."""
        dummy_u = jnp.ones(u_shape)
        dummy_y = jnp.ones(y_shape)
        params = self.init(rng, dummy_u, dummy_y, network)['params']
        
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=tx
        )
    
    def fit(self, train_data: Dict[str, jnp.ndarray], network: PhotonicNetwork,
            epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> Dict[str, Any]:
        """Train the quantum DeepONet."""
        rng = jax.random.PRNGKey(42)
        
        # Initialize training state
        u_shape = (batch_size, train_data['u'].shape[-1])
        y_shape = (batch_size, *train_data['y'].shape[1:])
        state = self.create_train_state(rng, u_shape, y_shape, network, lr)
        
        @jax.jit
        def train_step(state, batch_u, batch_y, batch_s):
            def loss_fn(params):
                predictions = self.apply({'params': params}, batch_u, batch_y, network)
                return jnp.mean((predictions - batch_s) ** 2)
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        # Training loop
        losses = []
        n_samples = len(train_data['u'])
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            perm = jax.random.permutation(rng, n_samples)
            u_shuffled = train_data['u'][perm]
            y_shuffled = train_data['y'][perm] 
            s_shuffled = train_data['s'][perm]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_u = u_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                batch_s = s_shuffled[start_idx:end_idx]
                
                state, loss = train_step(state, batch_u, batch_y, batch_s)
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
        
        predictions = self.apply(
            {'params': self.trained_params}, 
            test_data['u'], test_data['y'], network
        )
        return predictions