"""Input validation and error handling utilities for QNet-NO."""

from typing import Any, Dict, List, Optional, Tuple, Union
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class WarningLevel(Enum):
    """Warning severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[Tuple[str, WarningLevel]]
    suggestions: List[str]


def validate_tensor_shape(tensor: jnp.ndarray, expected_shape: Optional[Tuple[int, ...]] = None,
                         min_dims: Optional[int] = None, max_dims: Optional[int] = None,
                         name: str = "tensor") -> ValidationResult:
    """
    Validate tensor shape and dimensions.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected exact shape (None values are wildcards)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        name: Name of tensor for error messages
        
    Returns:
        ValidationResult with validation status and messages
    """
    errors = []
    warnings = []
    suggestions = []
    
    # Check if tensor is valid
    if not isinstance(tensor, (jnp.ndarray, np.ndarray)):
        errors.append(f"{name} must be a JAX or NumPy array, got {type(tensor)}")
        return ValidationResult(False, errors, warnings, suggestions)
    
    # Check dimensions
    actual_dims = len(tensor.shape)
    
    if min_dims is not None and actual_dims < min_dims:
        errors.append(f"{name} has {actual_dims} dimensions, minimum required: {min_dims}")
    
    if max_dims is not None and actual_dims > max_dims:
        errors.append(f"{name} has {actual_dims} dimensions, maximum allowed: {max_dims}")
    
    # Check exact shape if provided
    if expected_shape is not None:
        if len(expected_shape) != actual_dims:
            errors.append(f"{name} shape {tensor.shape} doesn't match expected dimensions {len(expected_shape)}")
        else:
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None and actual != expected:
                    errors.append(f"{name} dimension {i}: got {actual}, expected {expected}")
    
    # Check for common issues
    if tensor.size == 0:
        errors.append(f"{name} is empty")
    
    if jnp.any(jnp.isnan(tensor)):
        errors.append(f"{name} contains NaN values")
    
    if jnp.any(jnp.isinf(tensor)):
        warnings.append((f"{name} contains infinite values", WarningLevel.HIGH))
        suggestions.append(f"Consider clipping {name} values to finite range")
    
    # Check tensor magnitude
    tensor_max = jnp.max(jnp.abs(tensor))
    if tensor_max > 1e6:
        warnings.append((f"{name} has very large values (max: {tensor_max:.2e})", WarningLevel.MEDIUM))
        suggestions.append(f"Consider normalizing {name} for numerical stability")
    
    if tensor_max < 1e-10:
        warnings.append((f"{name} has very small values (max: {tensor_max:.2e})", WarningLevel.MEDIUM))
        suggestions.append(f"Check if {name} scaling is appropriate")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, suggestions)


def validate_network_parameters(nodes: int, fidelity_threshold: float, 
                               topology: str) -> ValidationResult:
    """Validate quantum network parameters."""
    errors = []
    warnings = []
    suggestions = []
    
    # Validate number of nodes
    if not isinstance(nodes, int):
        errors.append(f"Number of nodes must be integer, got {type(nodes)}")
    elif nodes < 1:
        errors.append(f"Number of nodes must be positive, got {nodes}")
    elif nodes > 1024:
        errors.append(f"Number of nodes too large: {nodes} > 1024")
    elif nodes > 100:
        warnings.append((f"Large number of nodes ({nodes}) may be slow", WarningLevel.MEDIUM))
        suggestions.append("Consider using fewer nodes for faster computation")
    
    # Validate fidelity threshold
    if not isinstance(fidelity_threshold, (int, float)):
        errors.append(f"Fidelity threshold must be numeric, got {type(fidelity_threshold)}")
    elif not 0 <= fidelity_threshold <= 1:
        errors.append(f"Fidelity threshold must be in [0,1], got {fidelity_threshold}")
    elif fidelity_threshold < 0.5:
        warnings.append((f"Low fidelity threshold ({fidelity_threshold:.3f})", WarningLevel.HIGH))
        suggestions.append("Consider increasing fidelity threshold for better quantum advantage")
    elif fidelity_threshold > 0.99:
        warnings.append((f"Very high fidelity threshold ({fidelity_threshold:.3f})", WarningLevel.LOW))
        suggestions.append("Such high fidelity may not be achievable on real hardware")
    
    # Validate topology
    valid_topologies = {"complete", "ring", "star", "grid", "random"}
    if topology not in valid_topologies:
        errors.append(f"Invalid topology '{topology}', must be one of {valid_topologies}")
    
    # Topology-specific warnings
    if topology == "complete" and nodes > 20:
        warnings.append((f"Complete topology with {nodes} nodes creates many links", WarningLevel.MEDIUM))
        suggestions.append("Consider grid or ring topology for large networks")
    
    if topology == "ring" and nodes < 3:
        warnings.append((f"Ring topology with {nodes} nodes is inefficient", WarningLevel.LOW))
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, suggestions)


def validate_training_parameters(epochs: int, batch_size: int, learning_rate: float,
                                data_size: int) -> ValidationResult:
    """Validate neural network training parameters."""
    errors = []
    warnings = []
    suggestions = []
    
    # Validate epochs
    if not isinstance(epochs, int):
        errors.append(f"Epochs must be integer, got {type(epochs)}")
    elif epochs < 1:
        errors.append(f"Epochs must be positive, got {epochs}")
    elif epochs > 10000:
        warnings.append((f"Very large number of epochs ({epochs})", WarningLevel.MEDIUM))
        suggestions.append("Consider using early stopping or reducing epochs")
    
    # Validate batch size
    if not isinstance(batch_size, int):
        errors.append(f"Batch size must be integer, got {type(batch_size)}")
    elif batch_size < 1:
        errors.append(f"Batch size must be positive, got {batch_size}")
    elif batch_size > data_size:
        errors.append(f"Batch size ({batch_size}) larger than dataset ({data_size})")
    elif batch_size > data_size // 2:
        warnings.append((f"Large batch size ({batch_size}) relative to data size ({data_size})", WarningLevel.MEDIUM))
        suggestions.append("Consider reducing batch size for better gradient estimates")
    elif batch_size == 1:
        warnings.append(("Batch size of 1 may lead to unstable training", WarningLevel.HIGH))
        suggestions.append("Consider using larger batch size (8-64)")
    
    # Validate learning rate
    if not isinstance(learning_rate, (int, float)):
        errors.append(f"Learning rate must be numeric, got {type(learning_rate)}")
    elif learning_rate <= 0:
        errors.append(f"Learning rate must be positive, got {learning_rate}")
    elif learning_rate > 1.0:
        warnings.append((f"Very high learning rate ({learning_rate})", WarningLevel.HIGH))
        suggestions.append("High learning rates may cause training instability")
    elif learning_rate < 1e-6:
        warnings.append((f"Very low learning rate ({learning_rate:.2e})", WarningLevel.MEDIUM))
        suggestions.append("Low learning rates may lead to slow convergence")
    elif learning_rate > 0.1:
        warnings.append((f"High learning rate ({learning_rate})", WarningLevel.MEDIUM))
        suggestions.append("Consider using learning rate scheduling")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, suggestions)


def validate_quantum_circuit(gates: List[Dict[str, Any]], n_qubits: int,
                            measurements: Optional[List[int]] = None) -> ValidationResult:
    """Validate quantum circuit specification."""
    errors = []
    warnings = []
    suggestions = []
    
    if not isinstance(gates, list):
        errors.append(f"Gates must be a list, got {type(gates)}")
        return ValidationResult(False, errors, warnings, suggestions)
    
    if len(gates) == 0:
        warnings.append(("Empty circuit", WarningLevel.LOW))
    
    # Validate each gate
    for i, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(f"Gate {i} must be a dictionary, got {type(gate)}")
            continue
        
        if "gate" not in gate:
            errors.append(f"Gate {i} missing 'gate' field")
            continue
        
        gate_type = gate["gate"]
        
        # Check qubit indices
        qubits_used = []
        if "qubit" in gate:
            qubits_used.append(gate["qubit"])
        if "control" in gate:
            qubits_used.append(gate["control"])
        if "target" in gate:
            qubits_used.append(gate["target"])
        
        for qubit in qubits_used:
            if not isinstance(qubit, int):
                errors.append(f"Gate {i} qubit index must be integer, got {type(qubit)}")
            elif qubit < 0:
                errors.append(f"Gate {i} qubit index must be non-negative, got {qubit}")
            elif qubit >= n_qubits:
                errors.append(f"Gate {i} qubit index {qubit} >= n_qubits {n_qubits}")
        
        # Check for duplicate qubits in two-qubit gates
        if len(qubits_used) > 1 and len(set(qubits_used)) != len(qubits_used):
            errors.append(f"Gate {i} has duplicate qubit indices: {qubits_used}")
        
        # Gate-specific validation
        if gate_type in ["rx", "ry", "rz", "phase"] and "angle" not in gate:
            warnings.append((f"Gate {i} ({gate_type}) missing angle parameter", WarningLevel.MEDIUM))
        
        if gate_type == "cnot" and ("control" not in gate or "target" not in gate):
            errors.append(f"Gate {i} (CNOT) missing control or target")
    
    # Validate measurements
    if measurements is not None:
        if not isinstance(measurements, list):
            errors.append(f"Measurements must be a list, got {type(measurements)}")
        else:
            for i, qubit in enumerate(measurements):
                if not isinstance(qubit, int):
                    errors.append(f"Measurement {i} qubit must be integer, got {type(qubit)}")
                elif qubit < 0 or qubit >= n_qubits:
                    errors.append(f"Measurement {i} qubit {qubit} out of range [0, {n_qubits-1}]")
    
    # Circuit depth warning
    if len(gates) > 1000:
        warnings.append((f"Very deep circuit ({len(gates)} gates)", WarningLevel.MEDIUM))
        suggestions.append("Deep circuits may suffer from noise accumulation")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, suggestions)


def validate_pde_data(data: Dict[str, jnp.ndarray], expected_keys: List[str]) -> ValidationResult:
    """Validate PDE dataset structure and content."""
    errors = []
    warnings = []
    suggestions = []
    
    # Check required keys
    missing_keys = set(expected_keys) - set(data.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
    
    # Check data consistency
    if "inputs" in data and "targets" in data:
        input_shape = data["inputs"].shape
        target_shape = data["targets"].shape
        
        if input_shape[0] != target_shape[0]:
            errors.append(f"Input batch size {input_shape[0]} != target batch size {target_shape[0]}")
        
        # Check spatial dimensions compatibility
        if len(input_shape) >= 3 and len(target_shape) >= 3:
            input_spatial = input_shape[1:-1]
            target_spatial = target_shape[1:-1]
            
            if input_spatial != target_spatial:
                warnings.append((f"Input spatial shape {input_spatial} != target spatial shape {target_spatial}", 
                               WarningLevel.MEDIUM))
                suggestions.append("Consider reshaping data to match spatial dimensions")
    
    # Validate individual arrays
    for key, array in data.items():
        if key in expected_keys:
            result = validate_tensor_shape(array, name=f"data['{key}']")
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            suggestions.extend(result.suggestions)
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, suggestions)


def validate_operator_parameters(modes: Optional[int] = None, width: Optional[int] = None,
                                schmidt_rank: Optional[int] = None, n_layers: Optional[int] = None) -> ValidationResult:
    """Validate quantum neural operator parameters."""
    errors = []
    warnings = []
    suggestions = []
    
    # Validate modes
    if modes is not None:
        if not isinstance(modes, int):
            errors.append(f"Modes must be integer, got {type(modes)}")
        elif modes < 1:
            errors.append(f"Modes must be positive, got {modes}")
        elif modes > 256:
            warnings.append((f"Large number of modes ({modes})", WarningLevel.MEDIUM))
            suggestions.append("Large mode counts increase computational cost")
        elif modes & (modes - 1) != 0:  # Check if power of 2
            warnings.append((f"Modes ({modes}) not a power of 2", WarningLevel.LOW))
            suggestions.append("Powers of 2 are more efficient for FFT operations")
    
    # Validate width
    if width is not None:
        if not isinstance(width, int):
            errors.append(f"Width must be integer, got {type(width)}")
        elif width < 1:
            errors.append(f"Width must be positive, got {width}")
        elif width > 2048:
            warnings.append((f"Very large width ({width})", WarningLevel.MEDIUM))
            suggestions.append("Large widths increase memory usage significantly")
    
    # Validate Schmidt rank
    if schmidt_rank is not None:
        if not isinstance(schmidt_rank, int):
            errors.append(f"Schmidt rank must be integer, got {type(schmidt_rank)}")
        elif schmidt_rank < 1:
            errors.append(f"Schmidt rank must be positive, got {schmidt_rank}")
        elif schmidt_rank > 64:
            warnings.append((f"High Schmidt rank ({schmidt_rank})", WarningLevel.MEDIUM))
            suggestions.append("High Schmidt ranks may not provide additional benefits")
        elif schmidt_rank & (schmidt_rank - 1) != 0:
            warnings.append((f"Schmidt rank ({schmidt_rank}) not a power of 2", WarningLevel.LOW))
    
    # Validate layers
    if n_layers is not None:
        if not isinstance(n_layers, int):
            errors.append(f"Number of layers must be integer, got {type(n_layers)}")
        elif n_layers < 1:
            errors.append(f"Number of layers must be positive, got {n_layers}")
        elif n_layers > 20:
            warnings.append((f"Deep network ({n_layers} layers)", WarningLevel.MEDIUM))
            suggestions.append("Very deep networks may suffer from vanishing gradients")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, suggestions)


def log_validation_result(result: ValidationResult, operation: str):
    """Log validation results with appropriate severity levels."""
    if not result.is_valid:
        logger.error(f"Validation failed for {operation}:")
        for error in result.errors:
            logger.error(f"  ERROR: {error}")
    
    for warning, level in result.warnings:
        if level == WarningLevel.CRITICAL:
            logger.critical(f"  CRITICAL: {warning}")
        elif level == WarningLevel.HIGH:
            logger.warning(f"  WARNING: {warning}")
        elif level == WarningLevel.MEDIUM:
            logger.info(f"  INFO: {warning}")
        else:  # LOW
            logger.debug(f"  DEBUG: {warning}")
    
    if result.suggestions:
        logger.info(f"Suggestions for {operation}:")
        for suggestion in result.suggestions:
            logger.info(f"  - {suggestion}")


def safe_execute(func: callable, *args, operation_name: str = "operation", **kwargs):
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        operation_name: Name for logging
        **kwargs: Keyword arguments
        
    Returns:
        Result of function execution or None if failed
    """
    try:
        logger.debug(f"Starting {operation_name}")
        result = func(*args, **kwargs)
        logger.debug(f"Completed {operation_name} successfully")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in {operation_name}: {e}")
        raise
        
    except ValueError as e:
        logger.error(f"Value error in {operation_name}: {e}")
        raise ValidationError(f"Invalid input for {operation_name}: {e}")
        
    except MemoryError as e:
        logger.error(f"Memory error in {operation_name}: {e}")
        raise ValidationError(f"Insufficient memory for {operation_name}")
        
    except Exception as e:
        logger.error(f"Unexpected error in {operation_name}: {e}")
        raise ValidationError(f"Operation {operation_name} failed: {e}")


def check_quantum_advantage_feasibility(network_size: int, problem_size: int, 
                                       schmidt_rank: int) -> ValidationResult:
    """
    Check if quantum advantage is feasible for given parameters.
    
    Provides guidance on when quantum enhancement is expected to help.
    """
    errors = []
    warnings = []
    suggestions = []
    
    # Classical complexity estimate
    classical_ops = problem_size ** 2  # Simplified estimate
    
    # Quantum complexity estimate  
    quantum_ops = network_size * schmidt_rank * np.log2(problem_size)
    
    # Quantum advantage threshold
    advantage_ratio = classical_ops / quantum_ops if quantum_ops > 0 else 0
    
    if advantage_ratio < 1.1:
        warnings.append((f"Limited quantum advantage expected (ratio: {advantage_ratio:.2f})", 
                        WarningLevel.HIGH))
        suggestions.append("Consider increasing network size or Schmidt rank")
        suggestions.append("Try problems with higher dimensionality or complexity")
    
    elif advantage_ratio < 2.0:
        warnings.append((f"Modest quantum advantage expected (ratio: {advantage_ratio:.2f})", 
                        WarningLevel.MEDIUM))
        suggestions.append("Good candidate for quantum enhancement")
    
    else:
        suggestions.append(f"Strong quantum advantage expected (ratio: {advantage_ratio:.2f})")
    
    # Network utilization check
    if network_size < 4:
        warnings.append((f"Small network size ({network_size})", WarningLevel.MEDIUM))
        suggestions.append("Consider using more quantum nodes for better parallelization")
    
    # Schmidt rank utilization
    if schmidt_rank < 4:
        warnings.append((f"Low Schmidt rank ({schmidt_rank})", WarningLevel.LOW))
        suggestions.append("Higher Schmidt rank may provide more entanglement resources")
    
    is_valid = True  # This is an advisory check, not a strict validation
    return ValidationResult(is_valid, errors, warnings, suggestions)