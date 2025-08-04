"""Classical neural network layers for hybrid quantum-classical models."""

from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class ResidualBlock(nn.Module):
    """
    Residual block for deep neural networks.
    
    Implements skip connections to enable training of very deep networks
    and improve gradient flow in hybrid quantum-classical architectures.
    """
    
    features: int
    activation: str = "gelu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Store input for skip connection
        residual = x
        
        # First layer
        x = nn.Dense(self.features)(x)
        
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = self._apply_activation(x)
        
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Second layer
        x = nn.Dense(self.features)(x)
        
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Skip connection
        if residual.shape[-1] != self.features:
            # Project residual to match dimensions
            residual = nn.Dense(self.features)(residual)
        
        x = x + residual
        x = self._apply_activation(x)
        
        return x
    
    def _apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply specified activation function."""
        if self.activation == "relu":
            return nn.relu(x)
        elif self.activation == "gelu":
            return nn.gelu(x)
        elif self.activation == "tanh":
            return nn.tanh(x)
        elif self.activation == "swish":
            return nn.swish(x)
        elif self.activation == "elu":
            return nn.elu(x)
        else:
            return x  # Linear activation


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer.
    
    Enables the model to focus on different parts of the input sequence
    and capture long-range dependencies in quantum neural operator tasks.
    """
    
    embed_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    use_bias: bool = True
    
    def setup(self):
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.query_projection = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.key_projection = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.value_projection = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        self.output_projection = nn.Dense(self.embed_dim, use_bias=self.use_bias)
        
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 training: bool = True) -> jnp.ndarray:
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute queries, keys, and values
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        # Reshape for multi-head attention
        queries = self._reshape_for_attention(queries, batch_size, seq_len)
        keys = self._reshape_for_attention(keys, batch_size, seq_len)
        values = self._reshape_for_attention(values, batch_size, seq_len)
        
        # Compute attention
        attention_output = self._scaled_dot_product_attention(
            queries, keys, values, mask, training
        )
        
        # Reshape back and apply output projection
        attention_output = attention_output.reshape(batch_size, seq_len, embed_dim)
        output = self.output_projection(attention_output)
        
        return output
    
    def _reshape_for_attention(self, x: jnp.ndarray, batch_size: int, seq_len: int) -> jnp.ndarray:
        """Reshape tensor for multi-head attention computation."""
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    
    def _scaled_dot_product_attention(self, queries: jnp.ndarray, keys: jnp.ndarray,
                                    values: jnp.ndarray, mask: Optional[jnp.ndarray],
                                    training: bool) -> jnp.ndarray:
        """Compute scaled dot-product attention."""
        # Compute attention scores
        scores = jnp.matmul(queries, keys.transpose(0, 1, 3, 2))
        scores = scores / jnp.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            scores = jnp.where(mask, scores, -jnp.inf)
        
        # Apply softmax
        attention_weights = nn.softmax(scores, axis=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, deterministic=not training)
        
        # Apply attention to values
        attention_output = jnp.matmul(attention_weights, values)
        
        return attention_output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two-layer MLP with configurable activation function,
    commonly used in transformer architectures.
    """
    
    hidden_dim: int
    output_dim: Optional[int] = None
    activation: str = "gelu"
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        output_dim = self.output_dim or x.shape[-1]
        
        # First layer
        x = nn.Dense(self.hidden_dim)(x)
        x = self._apply_activation(x)
        
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Second layer
        x = nn.Dense(output_dim)(x)
        
        return x
    
    def _apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply specified activation function."""
        if self.activation == "relu":
            return nn.relu(x)
        elif self.activation == "gelu":
            return nn.gelu(x)
        elif self.activation == "swish":
            return nn.swish(x)
        elif self.activation == "tanh":
            return nn.tanh(x)
        else:
            return x


class LayerNormalization(nn.Module):
    """
    Layer normalization for stabilizing training.
    
    Normalizes inputs across the feature dimension,
    helping with gradient flow and training stability.
    """
    
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feature_dim = x.shape[-1]
        
        # Compute mean and variance along feature dimension
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        
        # Normalize
        normalized = (x - mean) / jnp.sqrt(variance + self.epsilon)
        
        # Apply learnable scale and bias
        if self.use_scale:
            scale = self.param('scale', nn.initializers.ones, (feature_dim,))
            normalized = normalized * scale
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (feature_dim,))
            normalized = normalized + bias
        
        return normalized


class ConvolutionalBlock(nn.Module):
    """
    Convolutional block for spatial feature extraction.
    
    Combines convolution, normalization, and activation with
    optional residual connections for computer vision tasks.
    """
    
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"
    activation: str = "gelu"
    use_batch_norm: bool = True
    use_residual: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        input_features = x.shape[-1]
        residual = x
        
        # Convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding
        )(x)
        
        # Batch normalization
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Activation
        x = self._apply_activation(x)
        
        # Dropout
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Residual connection
        if self.use_residual and input_features == self.features and self.strides == (1, 1):
            x = x + residual
        elif self.use_residual:
            # Project residual to match dimensions
            residual_proj = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                padding="SAME"
            )(residual)
            x = x + residual_proj
        
        return x
    
    def _apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply specified activation function."""
        if self.activation == "relu":
            return nn.relu(x)
        elif self.activation == "gelu":
            return nn.gelu(x)
        elif self.activation == "swish":
            return nn.swish(x)
        elif self.activation == "tanh":
            return nn.tanh(x)
        elif self.activation == "leaky_relu":
            return nn.leaky_relu(x, negative_slope=0.01)
        else:
            return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence models.
    
    Adds position information to input embeddings using
    sine and cosine functions of different frequencies.
    """
    
    max_length: int = 10000
    
    def setup(self):
        # Pre-compute positional encodings
        self.positional_encodings = self._create_positional_encodings()
    
    def _create_positional_encodings(self) -> jnp.ndarray:
        """Create sinusoidal positional encodings."""
        # This will be set when the module is called for the first time
        return None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, embed_dim = x.shape
        
        # Create positional encodings if not already created
        if self.positional_encodings is None:
            pe = jnp.zeros((self.max_length, embed_dim))
            
            position = jnp.arange(0, self.max_length)[:, None]
            div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * -(jnp.log(10000.0) / embed_dim))
            
            pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
            pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
            
            self.positional_encodings = pe
        
        # Add positional encodings to input
        return x + self.positional_encodings[:seq_len, :]


class TransformerBlock(nn.Module):
    """
    Complete transformer block with attention and feed-forward layers.
    
    Combines multi-head attention, layer normalization, and feed-forward
    processing with residual connections.
    """
    
    embed_dim: int
    num_heads: int = 8
    hidden_dim: Optional[int] = None
    dropout_rate: float = 0.1
    activation: str = "gelu"
    
    def setup(self):
        self.hidden_dim = self.hidden_dim or 4 * self.embed_dim
        
        self.attention = AttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        
        self.ffn = FeedForwardNetwork(
            hidden_dim=self.hidden_dim,
            output_dim=self.embed_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 training: bool = True) -> jnp.ndarray:
        # Multi-head attention with residual connection
        attention_output = self.attention(x, mask=mask, training=training)
        x = self.norm1(x + attention_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + ffn_output)
        
        return x


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling layer for variable-size inputs.
    
    Reduces spatial dimensions to fixed output size regardless
    of input dimensions, useful for handling variable-size PDEs.
    """
    
    output_size: tuple
    pooling_type: str = "avg"  # "avg", "max", "adaptive_avg"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        input_spatial = x.shape[1:-1]  # Exclude batch and channel dimensions
        channels = x.shape[-1]
        
        if self.pooling_type == "avg":
            # Simple average pooling with computed kernel size
            kernel_sizes = tuple(
                input_dim // output_dim 
                for input_dim, output_dim in zip(input_spatial, self.output_size)
            )
            
            if len(input_spatial) == 1:
                pooled = nn.avg_pool(x, window_shape=(kernel_sizes[0],), strides=(kernel_sizes[0],))
            elif len(input_spatial) == 2:
                pooled = nn.avg_pool(x, window_shape=kernel_sizes, strides=kernel_sizes)
            else:
                # For higher dimensions, use global average pooling then reshape
                pooled = jnp.mean(x, axis=tuple(range(1, len(input_spatial) + 1)), keepdims=True)
                pooled = jnp.broadcast_to(pooled, (batch_size,) + self.output_size + (channels,))
        
        elif self.pooling_type == "max":
            kernel_sizes = tuple(
                input_dim // output_dim 
                for input_dim, output_dim in zip(input_spatial, self.output_size)
            )
            
            if len(input_spatial) == 1:
                pooled = nn.max_pool(x, window_shape=(kernel_sizes[0],), strides=(kernel_sizes[0],))
            elif len(input_spatial) == 2:
                pooled = nn.max_pool(x, window_shape=kernel_sizes, strides=kernel_sizes)
            else:
                pooled = jnp.max(x, axis=tuple(range(1, len(input_spatial) + 1)), keepdims=True)
                pooled = jnp.broadcast_to(pooled, (batch_size,) + self.output_size + (channels,))
        
        else:  # adaptive_avg
            # Adaptive average pooling using interpolation
            pooled = self._adaptive_avg_pool(x, self.output_size)
        
        return pooled
    
    def _adaptive_avg_pool(self, x: jnp.ndarray, output_size: tuple) -> jnp.ndarray:
        """Implement adaptive average pooling using interpolation."""
        # Simple implementation using jax.image.resize
        # In practice, would use more sophisticated adaptive pooling
        
        batch_size, channels = x.shape[0], x.shape[-1]
        target_shape = (batch_size,) + output_size + (channels,)
        
        # Use JAX's resize function for adaptive pooling
        pooled = jax.image.resize(x, target_shape, method="linear")
        
        return pooled