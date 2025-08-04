"""Base quantum backend interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class QuantumCircuit:
    """Represents a quantum circuit to be executed."""
    gates: List[Dict[str, Any]]
    n_qubits: int
    measurements: Optional[List[int]] = None
    

@dataclass
class QuantumResult:
    """Result from quantum circuit execution."""
    measurement_counts: Optional[Dict[str, int]] = None
    statevector: Optional[jnp.ndarray] = None
    expectation_values: Optional[Dict[str, float]] = None
    fidelity: Optional[float] = None
    execution_time: Optional[float] = None
    

class QuantumBackend(ABC):
    """
    Abstract base class for quantum computing backends.
    
    Defines the interface that all quantum backends must implement
    for integration with the QNet-NO library.
    """
    
    def __init__(self, name: str, n_qubits: int, **kwargs):
        self.name = name
        self.n_qubits = n_qubits
        self.config = kwargs
        self.is_connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the quantum backend."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the quantum backend."""
        pass
    
    @abstractmethod
    def execute_circuit(self, circuit: QuantumCircuit, 
                       shots: int = 1024) -> QuantumResult:
        """Execute a quantum circuit on the backend."""
        pass
    
    @abstractmethod
    def get_backend_properties(self) -> Dict[str, Any]:
        """Get backend-specific properties and capabilities."""
        pass
    
    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Check if backend supports a specific quantum operation."""
        pass
    
    def create_entangled_state(self, qubits: List[int], 
                              entanglement_type: str = "bell") -> QuantumCircuit:
        """Create entangled state between specified qubits."""
        gates = []
        
        if entanglement_type == "bell":
            # Create Bell state between pairs
            for i in range(0, len(qubits), 2):
                if i + 1 < len(qubits):
                    gates.append({"gate": "h", "qubit": qubits[i]})
                    gates.append({"gate": "cnot", "control": qubits[i], "target": qubits[i + 1]})
        
        elif entanglement_type == "ghz":
            # Create GHZ state across all qubits
            if qubits:
                gates.append({"gate": "h", "qubit": qubits[0]})
                for i in range(1, len(qubits)):
                    gates.append({"gate": "cnot", "control": qubits[0], "target": qubits[i]})
        
        return QuantumCircuit(gates=gates, n_qubits=self.n_qubits)
    
    def measure_fidelity(self, target_state: jnp.ndarray, 
                        measured_state: jnp.ndarray) -> float:
        """Calculate fidelity between target and measured quantum states."""
        # State fidelity for pure states
        overlap = jnp.abs(jnp.vdot(target_state, measured_state)) ** 2
        return float(overlap)
    
    def estimate_gate_error(self, gate_type: str) -> float:
        """Estimate error rate for specific gate type."""
        # Default error rates - subclasses should override with real values
        default_errors = {
            "x": 0.001,
            "y": 0.001, 
            "z": 0.001,
            "h": 0.002,
            "cnot": 0.01,
            "cz": 0.01,
            "measurement": 0.05,
        }
        return default_errors.get(gate_type, 0.01)
    
    def get_coherence_time(self) -> float:
        """Get typical coherence time in microseconds."""
        # Default value - subclasses should override
        return 100.0
    
    def get_gate_time(self, gate_type: str) -> float:
        """Get typical gate execution time in microseconds."""
        # Default values - subclasses should override
        default_times = {
            "x": 0.1,
            "y": 0.1,
            "z": 0.01,  # Virtual Z gate
            "h": 0.1,
            "cnot": 0.5,
            "cz": 0.3,
            "measurement": 1.0,
        }
        return default_times.get(gate_type, 0.1)
    
    def validate_circuit(self, circuit: QuantumCircuit) -> Tuple[bool, List[str]]:
        """Validate that circuit can be executed on this backend."""
        errors = []
        
        # Check qubit count
        if circuit.n_qubits > self.n_qubits:
            errors.append(f"Circuit requires {circuit.n_qubits} qubits, backend has {self.n_qubits}")
        
        # Check gate support
        for gate in circuit.gates:
            gate_type = gate.get("gate", "unknown")
            if not self.supports_operation(gate_type):
                errors.append(f"Gate '{gate_type}' not supported by backend")
        
        # Check qubit indices
        for gate in circuit.gates:
            qubits_used = []
            if "qubit" in gate:
                qubits_used.append(gate["qubit"])
            if "control" in gate:
                qubits_used.append(gate["control"])
            if "target" in gate:
                qubits_used.append(gate["target"])
            
            for qubit in qubits_used:
                if qubit >= self.n_qubits:
                    errors.append(f"Qubit index {qubit} out of range (max: {self.n_qubits - 1})")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', n_qubits={self.n_qubits})"