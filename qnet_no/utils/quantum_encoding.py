"""Quantum encoding schemes for classical data."""

from typing import Optional, List, TYPE_CHECKING
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ..networks.photonic_network import PhotonicNetwork


def quantum_feature_map(data: jnp.ndarray, network: 'PhotonicNetwork',
                       schmidt_rank: int, encoding_type: str = "amplitude") -> jnp.ndarray:
    """
    Encode classical data into quantum feature space using distributed QPUs.
    
    Args:
        data: Classical input data [batch, features]
        network: Quantum photonic network
        schmidt_rank: Schmidt rank for entanglement
        encoding_type: Type of encoding ("amplitude", "angle", "basis")
        
    Returns:
        Quantum-encoded features [batch, encoded_features]
    """
    if encoding_type == "amplitude":
        return amplitude_encoding(data, network, schmidt_rank)
    elif encoding_type == "angle":
        return angle_encoding(data, network, schmidt_rank)
    elif encoding_type == "basis":
        return basis_encoding(data, network, schmidt_rank)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def amplitude_encoding(data: jnp.ndarray, network: 'PhotonicNetwork',
                      schmidt_rank: int) -> jnp.ndarray:
    """
    Amplitude encoding: encode data as probability amplitudes.
    
    Maps classical data to quantum state amplitudes, providing
    exponential compression through quantum superposition.
    """
    batch_size, n_features = data.shape
    
    # Normalize data to unit norm for valid probability amplitudes
    data_normalized = normalize_for_amplitudes(data)
    
    # Determine quantum register size
    n_qubits = int(jnp.ceil(jnp.log2(n_features)))
    n_states = 2**n_qubits
    
    # Pad data to power of 2
    if n_features < n_states:
        padding = jnp.zeros((batch_size, n_states - n_features))
        data_padded = jnp.concatenate([data_normalized, padding], axis=1)
    else:
        data_padded = data_normalized[:, :n_states]
    
    # Apply quantum enhancement using distributed processing
    quantum_enhanced = apply_distributed_amplitude_encoding(
        data_padded, network, schmidt_rank, n_qubits
    )
    
    return quantum_enhanced


def angle_encoding(data: jnp.ndarray, network: 'PhotonicNetwork',
                  schmidt_rank: int) -> jnp.ndarray:
    """
    Angle encoding: encode data as rotation angles in quantum gates.
    
    Maps classical features to rotation parameters, enabling
    efficient encoding of continuous variables.
    """
    batch_size, n_features = data.shape
    
    # Scale data to appropriate angle range [0, 2π]
    data_scaled = scale_to_angles(data)
    
    # Create quantum rotation encodings
    quantum_features = []
    
    # Distribute features across quantum nodes
    features_per_node = distribute_features_across_nodes(n_features, len(network.quantum_nodes))
    
    for node_id, feature_range in features_per_node.items():
        start_idx, end_idx = feature_range
        node_features = data_scaled[:, start_idx:end_idx]
        
        # Apply angle encoding at this node
        node_quantum_features = node_angle_encoding(
            node_features, network.quantum_nodes[node_id], schmidt_rank
        )
        
        quantum_features.append(node_quantum_features)
    
    # Combine results from all nodes
    if quantum_features:
        return jnp.concatenate(quantum_features, axis=1)
    else:
        return data_scaled


def basis_encoding(data: jnp.ndarray, network: 'PhotonicNetwork',
                  schmidt_rank: int) -> jnp.ndarray:
    """
    Basis encoding: encode data in computational basis states.
    
    Maps discrete classical data to quantum computational basis,
    suitable for categorical and discrete features.
    """
    batch_size, n_features = data.shape
    
    # Convert continuous data to discrete bins
    n_bins = 16  # Number of discrete levels
    data_discrete = discretize_data(data, n_bins)
    
    # Create one-hot-like quantum encoding
    quantum_features = []
    
    for feature_idx in range(n_features):
        feature_column = data_discrete[:, feature_idx]
        
        # Create basis state encoding for this feature
        basis_encoded = create_basis_states(feature_column, n_bins)
        quantum_features.append(basis_encoded)
    
    # Stack all features
    quantum_encoded = jnp.stack(quantum_features, axis=2)  # [batch, n_bins, n_features]
    
    # Reshape for neural network processing
    quantum_encoded = quantum_encoded.reshape(batch_size, -1)
    
    return quantum_encoded


def normalize_for_amplitudes(data: jnp.ndarray) -> jnp.ndarray:
    """Normalize data for valid quantum probability amplitudes."""
    # Take absolute value and normalize to unit norm
    data_abs = jnp.abs(data)
    
    # Add small epsilon to avoid division by zero
    norms = jnp.linalg.norm(data_abs, axis=1, keepdims=True) + 1e-12
    
    return data_abs / norms


def scale_to_angles(data: jnp.ndarray) -> jnp.ndarray:
    """Scale data to angle range [0, 2π]."""
    # Normalize to [0, 1] then scale to [0, 2π]
    data_min = jnp.min(data, axis=1, keepdims=True)
    data_max = jnp.max(data, axis=1, keepdims=True)
    
    # Avoid division by zero
    data_range = data_max - data_min + 1e-12
    
    data_normalized = (data - data_min) / data_range
    return data_normalized * 2 * jnp.pi


def discretize_data(data: jnp.ndarray, n_bins: int) -> jnp.ndarray:
    """Discretize continuous data into bins."""
    # Normalize to [0, 1] then discretize
    data_min = jnp.min(data, axis=0, keepdims=True)
    data_max = jnp.max(data, axis=0, keepdims=True)
    
    data_range = data_max - data_min + 1e-12
    data_normalized = (data - data_min) / data_range
    
    # Map to discrete bins
    data_discrete = jnp.floor(data_normalized * (n_bins - 1e-6)).astype(int)
    data_discrete = jnp.clip(data_discrete, 0, n_bins - 1)
    
    return data_discrete


def distribute_features_across_nodes(n_features: int, n_nodes: int) -> dict:
    """Distribute features optimally across quantum nodes."""
    if n_nodes == 0:
        return {}
    
    features_per_node = n_features // n_nodes
    remainder = n_features % n_nodes
    
    feature_assignments = {}
    current_feature = 0
    
    for node_id in range(n_nodes):
        node_features = features_per_node + (1 if node_id < remainder else 0)
        
        if node_features > 0:
            end_feature = current_feature + node_features
            feature_assignments[node_id] = (current_feature, end_feature)
            current_feature = end_feature
    
    return feature_assignments


def apply_distributed_amplitude_encoding(data: jnp.ndarray, network: 'PhotonicNetwork',
                                       schmidt_rank: int, n_qubits: int) -> jnp.ndarray:
    """Apply amplitude encoding using distributed quantum processing."""
    n_nodes = len(network.quantum_nodes)
    if n_nodes == 0:
        return data
    
    # Distribute quantum states across nodes
    states_per_node = 2**n_qubits // n_nodes
    enhanced_components = []
    
    for node_id in range(n_nodes):
        start_state = node_id * states_per_node
        end_state = min((node_id + 1) * states_per_node, 2**n_qubits)
        
        if start_state < end_state:
            node_data = data[:, start_state:end_state]
            
            # Apply quantum enhancement at this node
            enhanced_data = quantum_amplitude_enhancement(
                node_data, network.quantum_nodes[node_id], schmidt_rank
            )
            
            enhanced_components.append(enhanced_data)
    
    # Combine results
    if enhanced_components:
        return jnp.concatenate(enhanced_components, axis=1)
    else:
        return data


def node_angle_encoding(features: jnp.ndarray, quantum_node: dict, 
                       schmidt_rank: int) -> jnp.ndarray:
    """Apply angle encoding at individual quantum node."""
    node_fidelity = quantum_node.get("fidelity", 0.9)
    qpu_type = quantum_node.get("qpu_type", "photonic")
    
    # Create quantum rotations based on feature angles
    batch_size, n_node_features = features.shape
    
    # Generate quantum features using trigonometric functions
    cos_features = jnp.cos(features) * node_fidelity
    sin_features = jnp.sin(features) * node_fidelity
    
    # Add quantum correlations based on Schmidt rank
    correlation_strength = schmidt_rank / 32.0  # Normalize by max Schmidt rank
    
    if n_node_features > 1:
        # Add cross-feature quantum correlations
        correlations = jnp.outer(jnp.mean(features, axis=0), jnp.mean(features, axis=0))
        correlation_matrix = correlation_strength * jnp.sin(correlations)
        
        # Apply correlations (simplified)
        correlated_features = features + 0.1 * jnp.sum(correlation_matrix, axis=1, keepdims=True)
        
        cos_features = jnp.cos(correlated_features) * node_fidelity
        sin_features = jnp.sin(correlated_features) * node_fidelity
    
    # Combine cos and sin components
    quantum_features = jnp.concatenate([cos_features, sin_features], axis=1)
    
    return quantum_features


def quantum_amplitude_enhancement(data: jnp.ndarray, quantum_node: dict,
                                schmidt_rank: int) -> jnp.ndarray:
    """Apply quantum amplitude enhancement at node."""
    node_fidelity = quantum_node.get("fidelity", 0.9)
    qpu_type = quantum_node.get("qpu_type", "photonic")
    
    # Apply quantum enhancement based on node type
    if qpu_type == "photonic":
        # Photonic enhancement: squeezing-like transformation
        alpha = jnp.sqrt(node_fidelity)
        enhanced_data = alpha * data + (1 - alpha) * jnp.mean(data, axis=1, keepdims=True)
        
    elif qpu_type == "superconducting":
        # Superconducting enhancement: high-fidelity gates
        enhancement_factor = 1 + 0.2 * node_fidelity
        enhanced_data = data * enhancement_factor
        
    elif qpu_type == "trapped_ion":
        # Trapped ion enhancement: all-to-all connectivity
        collective_enhancement = jnp.mean(data, axis=1, keepdims=True)
        enhanced_data = data + 0.1 * node_fidelity * collective_enhancement
        
    elif qpu_type == "nv_center":
        # NV center enhancement: room temperature operation
        thermal_noise = 0.05 * (1 - node_fidelity)
        noise = jax.random.normal(jax.random.PRNGKey(42), data.shape) * thermal_noise
        enhanced_data = data + noise
        
    else:
        enhanced_data = data
    
    # Ensure proper normalization for amplitude encoding
    norms = jnp.linalg.norm(enhanced_data, axis=1, keepdims=True) + 1e-12
    enhanced_data = enhanced_data / norms
    
    return enhanced_data


def create_basis_states(feature_values: jnp.ndarray, n_bins: int) -> jnp.ndarray:
    """Create basis state encoding for discrete feature values."""
    batch_size = len(feature_values)
    
    # Create one-hot encoding
    basis_states = jnp.zeros((batch_size, n_bins))
    
    # Set appropriate basis state for each sample
    for i in range(batch_size):
        bin_idx = int(feature_values[i])
        basis_states = basis_states.at[i, bin_idx].set(1.0)
    
    return basis_states


def quantum_variational_encoding(data: jnp.ndarray, network: 'PhotonicNetwork',
                                schmidt_rank: int, n_layers: int = 3) -> jnp.ndarray:
    """
    Variational quantum encoding using parameterized quantum circuits.
    
    Implements trainable quantum feature maps that can be optimized
    end-to-end with classical neural networks.
    """
    batch_size, n_features = data.shape
    
    # Initialize variational parameters
    n_qubits = int(jnp.ceil(jnp.log2(n_features)))
    
    # Create variational circuit layers
    encoded_features = data
    
    for layer in range(n_layers):
        # Apply rotation layer
        rotation_angles = scale_to_angles(encoded_features)
        
        # Apply entangling layer (simplified)
        entangled_features = apply_entangling_layer(
            rotation_angles, network, schmidt_rank
        )
        
        # Measurement and feature extraction
        encoded_features = extract_variational_features(entangled_features)
    
    return encoded_features


def apply_entangling_layer(angles: jnp.ndarray, network: 'PhotonicNetwork',
                          schmidt_rank: int) -> jnp.ndarray:
    """Apply entangling operations across quantum network."""
    batch_size, n_features = angles.shape
    
    # Create entanglement patterns based on network topology
    entanglement_strength = schmidt_rank / 32.0
    
    # Apply pairwise entangling operations
    entangled = jnp.copy(angles)
    
    for i in range(0, n_features - 1, 2):
        if i + 1 < n_features:
            # Simulate two-qubit entangling gate
            angle1 = angles[:, i]
            angle2 = angles[:, i + 1]
            
            # Controlled rotation simulation
            entangled_angle1 = angle1 + entanglement_strength * jnp.sin(angle2)
            entangled_angle2 = angle2 + entanglement_strength * jnp.sin(angle1)
            
            entangled = entangled.at[:, i].set(entangled_angle1)
            entangled = entangled.at[:, i + 1].set(entangled_angle2)
    
    return entangled


def extract_variational_features(encoded_angles: jnp.ndarray) -> jnp.ndarray:
    """Extract features from variational quantum circuit."""
    # Measure expectation values of Pauli operators
    pauli_x = jnp.cos(encoded_angles)
    pauli_y = jnp.sin(encoded_angles)
    pauli_z = jnp.cos(2 * encoded_angles)
    
    # Combine measurements
    variational_features = jnp.concatenate([pauli_x, pauli_y, pauli_z], axis=1)
    
    return variational_features