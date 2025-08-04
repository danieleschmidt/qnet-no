"""Photonic quantum computing backend using Strawberry Fields."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import jax.numpy as jnp
from .base_backend import QuantumBackend, QuantumCircuit, QuantumResult

try:
    import strawberryfields as sf
    from strawberryfields.ops import *
    STRAWBERRYFIELDS_AVAILABLE = True
except ImportError:
    STRAWBERRYFIELDS_AVAILABLE = False


class PhotonicBackend(QuantumBackend):
    """
    Photonic quantum computing backend using continuous variables.
    
    Interfaces with Xanadu's Strawberry Fields for photonic quantum
    computation including Gaussian operations and photon counting.
    """
    
    def __init__(self, name: str = "photonic", n_modes: int = 8, 
                 cutoff_dim: int = 10, **kwargs):
        super().__init__(name, n_modes, **kwargs)
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        self.engine = None
        self.program = None
        
        if not STRAWBERRYFIELDS_AVAILABLE:
            print("Warning: Strawberry Fields not available. Using simulation mode.")
    
    def connect(self) -> bool:
        """Connect to photonic quantum processor."""
        try:
            if STRAWBERRYFIELDS_AVAILABLE:
                # Create Strawberry Fields engine
                backend_name = self.config.get("backend", "fock")
                self.engine = sf.Engine(backend_name)
                self.is_connected = True
                return True
            else:
                # Simulation mode
                self.is_connected = True
                return True
        except Exception as e:
            print(f"Failed to connect to photonic backend: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from photonic quantum processor."""
        self.engine = None
        self.is_connected = False
    
    def execute_circuit(self, circuit: QuantumCircuit, 
                       shots: int = 1024) -> QuantumResult:
        """Execute quantum circuit on photonic processor."""
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Cannot connect to photonic backend")
        
        if STRAWBERRYFIELDS_AVAILABLE:
            return self._execute_sf_circuit(circuit, shots)
        else:
            return self._simulate_photonic_circuit(circuit, shots)
    
    def _execute_sf_circuit(self, circuit: QuantumCircuit, shots: int) -> QuantumResult:
        """Execute circuit using Strawberry Fields."""
        # Create Strawberry Fields program
        prog = sf.Program(self.n_modes)
        
        with prog.context as q:
            # Convert QNet-NO circuit to Strawberry Fields operations
            for gate in circuit.gates:
                self._apply_sf_gate(gate, q)
            
            # Add measurements if specified
            if circuit.measurements:
                for mode in circuit.measurements:
                    MeasureFock() | q[mode]
        
        # Execute program
        try:
            result = self.engine.run(prog, shots=shots)
            
            # Process results
            if hasattr(result, 'samples'):
                # Photon counting measurements
                samples = result.samples
                measurement_counts = {}
                
                for sample in samples:
                    key = ''.join(map(str, sample))
                    measurement_counts[key] = measurement_counts.get(key, 0) + 1
                
                return QuantumResult(measurement_counts=measurement_counts)
            
            elif hasattr(result, 'state'):
                # State-based result
                state = result.state
                statevector = jnp.array(state.ket()) if hasattr(state, 'ket') else None
                
                return QuantumResult(statevector=statevector)
            
            else:
                return QuantumResult()
                
        except Exception as e:
            print(f"Error executing Strawberry Fields circuit: {e}")
            return QuantumResult()
    
    def _apply_sf_gate(self, gate: Dict[str, Any], q: List) -> None:
        """Apply gate operation in Strawberry Fields format."""
        gate_type = gate.get("gate", "")
        
        if gate_type == "displacement":
            alpha = gate.get("alpha", 0.5)
            mode = gate.get("mode", 0)
            Dgate(alpha) | q[mode]
            
        elif gate_type == "squeezing":
            r = gate.get("r", 0.1)
            mode = gate.get("mode", 0)
            Sgate(r) | q[mode]
            
        elif gate_type == "rotation":
            phi = gate.get("phi", 0.0)
            mode = gate.get("mode", 0)
            Rgate(phi) | q[mode]
            
        elif gate_type == "beamsplitter":
            theta = gate.get("theta", np.pi/4)
            phi = gate.get("phi", 0.0)
            mode1 = gate.get("mode1", 0)
            mode2 = gate.get("mode2", 1)
            BSgate(theta, phi) | (q[mode1], q[mode2])
            
        elif gate_type == "two_mode_squeezing":
            r = gate.get("r", 0.1)
            phi = gate.get("phi", 0.0)
            mode1 = gate.get("mode1", 0)
            mode2 = gate.get("mode2", 1)
            S2gate(r, phi) | (q[mode1], q[mode2])
            
        elif gate_type == "kerr":
            kappa = gate.get("kappa", 0.01)
            mode = gate.get("mode", 0)
            Kgate(kappa) | q[mode]
    
    def _simulate_photonic_circuit(self, circuit: QuantumCircuit, shots: int) -> QuantumResult:
        """Simulate photonic circuit without Strawberry Fields."""
        # Simple simulation for testing purposes
        n_modes = circuit.n_qubits
        
        # Initialize vacuum state
        state = jnp.zeros(self.cutoff_dim ** n_modes, dtype=complex)
        state = state.at[0].set(1.0)  # Vacuum state |0,0,...,0>
        
        # Apply gates (simplified simulation)
        for gate in circuit.gates:
            state = self._simulate_photonic_gate(gate, state, n_modes)
        
        # Simulate measurements
        if circuit.measurements:
            measurement_counts = {}
            
            # Sample from state distribution
            probabilities = jnp.abs(state) ** 2
            
            for _ in range(shots):
                sample_idx = np.random.choice(len(probabilities), p=probabilities)
                # Convert index to Fock state representation
                fock_state = self._index_to_fock_state(sample_idx, n_modes)
                key = ''.join(map(str, fock_state))
                measurement_counts[key] = measurement_counts.get(key, 0) + 1
            
            return QuantumResult(measurement_counts=measurement_counts)
        
        return QuantumResult(statevector=state)
    
    def _simulate_photonic_gate(self, gate: Dict[str, Any], state: jnp.ndarray, 
                               n_modes: int) -> jnp.ndarray:
        """Simulate single photonic gate operation."""
        gate_type = gate.get("gate", "")
        
        if gate_type == "displacement":
            # Simplified displacement operation
            alpha = gate.get("alpha", 0.1)
            mode = gate.get("mode", 0)
            # Apply phase to simulate displacement (simplified)
            phase = np.angle(alpha)
            state = state * jnp.exp(1j * phase)
            
        elif gate_type == "beamsplitter":
            # Simplified beamsplitter (just apply phase)
            theta = gate.get("theta", np.pi/4)
            state = state * jnp.exp(1j * theta)
        
        # For other gates, return state unchanged (placeholder)
        return state
    
    def _index_to_fock_state(self, index: int, n_modes: int) -> List[int]:
        """Convert linear index to Fock state representation."""
        fock_state = []
        for _ in range(n_modes):
            fock_state.append(index % self.cutoff_dim)
            index //= self.cutoff_dim
        return fock_state
    
    def get_backend_properties(self) -> Dict[str, Any]:
        """Get photonic backend properties."""
        return {
            "backend_type": "photonic",
            "n_modes": self.n_modes,
            "cutoff_dimension": self.cutoff_dim,
            "supports_cv": True,
            "supports_fock": True,
            "native_gates": [
                "displacement", "squeezing", "rotation", 
                "beamsplitter", "two_mode_squeezing", "kerr"
            ],
            "measurement_types": ["homodyne", "heterodyne", "photon_counting"],
            "coherence_time": self.get_coherence_time(),
            "gate_fidelities": {
                "displacement": 0.99,
                "squeezing": 0.95,
                "beamsplitter": 0.98,
                "two_mode_squeezing": 0.92,
            }
        }
    
    def supports_operation(self, operation: str) -> bool:
        """Check if operation is supported by photonic backend."""
        supported_ops = {
            "displacement", "squeezing", "rotation", "beamsplitter",
            "two_mode_squeezing", "kerr", "homodyne", "heterodyne", "photon_counting"
        }
        return operation in supported_ops
    
    def get_coherence_time(self) -> float:
        """Get photonic coherence time (typically very short)."""
        return 0.1  # microseconds - photonic systems decohere quickly
    
    def get_gate_time(self, gate_type: str) -> float:
        """Get photonic gate execution times."""
        photonic_times = {
            "displacement": 0.001,  # Very fast
            "squeezing": 0.001,
            "rotation": 0.001,
            "beamsplitter": 0.005,
            "two_mode_squeezing": 0.01,
            "kerr": 0.1,
            "homodyne": 0.1,
            "heterodyne": 0.1,
            "photon_counting": 1.0,
        }
        return photonic_times.get(gate_type, 0.01)
    
    def create_gaussian_state(self, mean: jnp.ndarray, 
                             covariance: jnp.ndarray) -> QuantumCircuit:
        """Create Gaussian state preparation circuit."""
        gates = []
        n_modes = len(mean) // 2
        
        # Displacement operations
        for i in range(n_modes):
            x_mean = mean[2*i]
            p_mean = mean[2*i + 1]
            alpha = (x_mean + 1j * p_mean) / np.sqrt(2)
            
            if abs(alpha) > 1e-10:  # Only add if non-zero
                gates.append({
                    "gate": "displacement",
                    "mode": i,
                    "alpha": complex(alpha)
                })
        
        # Squeezing operations (simplified - would need full covariance decomposition)
        for i in range(n_modes):
            var_x = covariance[2*i, 2*i]
            var_p = covariance[2*i + 1, 2*i + 1]
            
            if var_x != var_p:  # Non-vacuum variance
                r = 0.5 * np.log(var_p / var_x) if var_x > 0 else 0
                if abs(r) > 1e-10:
                    gates.append({
                        "gate": "squeezing", 
                        "mode": i,
                        "r": float(r)
                    })
        
        return QuantumCircuit(gates=gates, n_qubits=n_modes)
    
    def measure_quadrature(self, mode: int, angle: float = 0.0) -> QuantumCircuit:
        """Create quadrature measurement circuit."""
        gates = []
        
        # Rotation before homodyne measurement
        if abs(angle) > 1e-10:
            gates.append({
                "gate": "rotation",
                "mode": mode, 
                "phi": float(angle)
            })
        
        # Homodyne measurement
        gates.append({
            "gate": "homodyne",
            "mode": mode,
            "angle": float(angle)
        })
        
        return QuantumCircuit(gates=gates, n_qubits=self.n_modes, measurements=[mode])