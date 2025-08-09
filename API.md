# QNet-NO API Reference

**Complete API documentation for the Hybrid Quantum-Classical Neural Operator framework**

## 🚀 Quick Start API Usage

```python
import qnet_no as qno
from qnet_no.algorithms import create_hybrid_scheduler
from qnet_no.operators import QuantumFourierNeuralOperator
from qnet_no.networks import PhotonicNetwork

# Create quantum network
network = PhotonicNetwork(nodes=8, topology='ring')

# Create hybrid scheduler
scheduler = create_hybrid_scheduler(network, qaoa_layers=4)

# Create quantum neural operator
qfno = QuantumFourierNeuralOperator(modes=16, schmidt_rank=8)

# Load data and train
data = qno.datasets.load_navier_stokes()
results = qfno.fit(data.train, network, epochs=100)
```

## 📚 Core API Modules

### `qnet_no.algorithms`

#### `HybridQuantumClassicalScheduler`

Main hybrid scheduling algorithm combining quantum optimization with classical heuristics.

```python
class HybridQuantumClassicalScheduler:
    def __init__(self, network: PhotonicNetwork, config: HybridSchedulingConfig = None)
    def schedule_tasks_hybrid(self, tasks: List[ComputationTask]) -> SchedulingResult
    def get_comprehensive_metrics(self) -> Dict[str, Any]
    def cleanup_resources(self) -> None
```

**Parameters:**
- `network`: PhotonicNetwork - The quantum photonic network
- `config`: HybridSchedulingConfig - Configuration parameters

**Returns:**
- `SchedulingResult` - Complete scheduling results with quantum advantage metrics

**Example:**
```python
from qnet_no.algorithms import HybridQuantumClassicalScheduler, HybridSchedulingConfig
from qnet_no.networks import PhotonicNetwork
from qnet_no.networks.entanglement_scheduler import ComputationTask, TaskPriority

# Setup
network = PhotonicNetwork(nodes=4)
config = HybridSchedulingConfig(qaoa_layers=4, adaptive_schmidt_rank=True)
scheduler = HybridQuantumClassicalScheduler(network, config)

# Create tasks
tasks = [
    ComputationTask(
        task_id="task_1",
        operation_type="fourier_transform", 
        required_qubits=4,
        estimated_time=100.0,
        priority=TaskPriority.HIGH
    )
]

# Schedule
result = scheduler.schedule_tasks_hybrid(tasks)
print(f"Quantum Advantage: {result.quantum_advantage_score:.2f}")
```

#### `AdaptiveSchmidtRankOptimizer`

Optimizes Schmidt rank dynamically based on problem complexity and network conditions.

```python
class AdaptiveSchmidtRankOptimizer:
    def __init__(self, min_rank: int = 2, max_rank: int = 64)
    def optimize_schmidt_rank(self, task: ComputationTask, network: PhotonicNetwork, 
                            current_performance: Dict[str, float]) -> int
```

**Example:**
```python
from qnet_no.algorithms import AdaptiveSchmidtRankOptimizer

optimizer = AdaptiveSchmidtRankOptimizer(min_rank=4, max_rank=32)
optimal_rank = optimizer.optimize_schmidt_rank(
    task=task,
    network=network,
    current_performance={"accuracy": 0.85, "throughput": 500.0}
)
```

#### Factory Functions

```python
def create_hybrid_scheduler(network: PhotonicNetwork, qaoa_layers: int = 4,
                          enable_adaptation: bool = True) -> HybridQuantumClassicalScheduler

def benchmark_quantum_advantage(network: PhotonicNetwork, tasks: List[ComputationTask],
                               n_trials: int = 10) -> Dict[str, Any]
```

### `qnet_no.operators`

#### `QuantumFourierNeuralOperator`

Quantum-enhanced Fourier Neural Operator for PDE solving.

```python
class QuantumFourierNeuralOperator(nn.Module):
    def __init__(self, modes: int = 16, width: int = 64, schmidt_rank: int = 8, n_layers: int = 4)
    def fit(self, train_data: Dict[str, jnp.ndarray], network: PhotonicNetwork,
            epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> Dict[str, Any]
    def predict(self, test_data: Dict[str, jnp.ndarray], network: PhotonicNetwork) -> jnp.ndarray
    def enable_distributed_computing(self, node_configs: list) -> None
    def auto_scale_batch_size(self, current_loss: float, memory_usage: float, throughput: float) -> int
```

**Parameters:**
- `modes`: int - Number of Fourier modes to retain
- `width`: int - Hidden layer width
- `schmidt_rank`: int - Quantum entanglement rank
- `n_layers`: int - Number of layers

**Example:**
```python
from qnet_no.operators import QuantumFourierNeuralOperator
from qnet_no.networks import PhotonicNetwork
import qnet_no as qno

# Create operator
qfno = QuantumFourierNeuralOperator(
    modes=16, 
    width=64,
    schmidt_rank=8,
    n_layers=4
)

# Create network
network = PhotonicNetwork(nodes=4, topology='complete')

# Load data
data = qno.datasets.load_heat_equation(resolution=64, n_samples=1000)

# Train
results = qfno.fit(
    train_data=data.train,
    network=network,
    epochs=100,
    lr=1e-3,
    batch_size=32
)

# Predict
predictions = qfno.predict(data.test, network)
print(f"Final loss: {results['losses'][-1]:.6f}")
print(f"Test shape: {predictions.shape}")
```

#### `QuantumDeepONet`

Quantum-enhanced DeepONet for operator learning.

```python
class QuantumDeepONet(nn.Module):
    def __init__(self, trunk_dim: int = 64, n_layers: int = 4, schmidt_rank: int = 8)
    def fit(self, train_data: Dict[str, jnp.ndarray], network: PhotonicNetwork,
            epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> Dict[str, Any]
    def predict(self, test_data: Dict[str, jnp.ndarray], network: PhotonicNetwork) -> jnp.ndarray
```

**Example:**
```python
from qnet_no.operators import QuantumDeepONet

# Create DeepONet
deeponet = QuantumDeepONet(
    trunk_dim=64,
    n_layers=4,
    schmidt_rank=8
)

# Train (requires function-location format data)
deeponet_data = qno.datasets.load_antiderivative_data()
results = deeponet.fit(deeponet_data.train, network)
```

### `qnet_no.networks`

#### `PhotonicNetwork`

Represents a distributed quantum photonic network.

```python
class PhotonicNetwork:
    def __init__(self, nodes: int = 4, topology: str = 'complete', fidelity_threshold: float = 0.8)
    def add_quantum_node(self, node_id: int, n_qubits: int, fidelity: float, 
                        capabilities: List[str], memory_gb: float = 4.0) -> None
    def add_entanglement_link(self, node1: int, node2: int, fidelity: float, schmidt_rank: int) -> None
    def get_entanglement_quality(self, node1: int, node2: int) -> Optional[float]
    def get_network_stats(self) -> Dict[str, Any]
```

**Example:**
```python
from qnet_no.networks import PhotonicNetwork

# Create network
network = PhotonicNetwork(nodes=6, topology='ring', fidelity_threshold=0.85)

# Add custom node
network.add_quantum_node(
    node_id=6,
    n_qubits=16,
    fidelity=0.95,
    capabilities=["two_qubit_gates", "readout", "parametric_ops"],
    memory_gb=8.0
)

# Add custom link
network.add_entanglement_link(
    node1=0, 
    node2=6,
    fidelity=0.90,
    schmidt_rank=16
)

# Get statistics
stats = network.get_network_stats()
print(f"Network has {stats['num_nodes']} nodes and {stats['num_links']} links")
```

#### `EntanglementScheduler`

Classical entanglement-aware scheduler (baseline comparison).

```python
class EntanglementScheduler:
    def __init__(self, network: PhotonicNetwork)
    def schedule_tasks(self, tasks: List[ComputationTask]) -> SchedulingResult
    def get_scheduling_metrics(self) -> Dict[str, Any]
```

### `qnet_no.datasets`

#### Data Loading Functions

```python
def load_navier_stokes(resolution: int = 64, n_samples: int = 1000, 
                      n_timesteps: int = 10) -> DataContainer

def load_heat_equation(resolution: int = 64, n_samples: int = 1000,
                      boundary_condition: str = 'periodic') -> DataContainer

def load_wave_equation(resolution: int = 64, n_samples: int = 1000,
                      wave_speed: float = 1.0) -> DataContainer

def generate_synthetic_pde_data(equation_type: str, resolution: int = 64,
                               n_samples: int = 1000) -> DataContainer
```

**Example:**
```python
import qnet_no as qno

# Load different PDE datasets
navier_stokes_data = qno.datasets.load_navier_stokes(
    resolution=128, 
    n_samples=5000,
    n_timesteps=20
)

heat_data = qno.datasets.load_heat_equation(
    resolution=64,
    n_samples=2000,
    boundary_condition='dirichlet'
)

# Generate synthetic data
synthetic_data = qno.datasets.generate_synthetic_pde_data(
    equation_type="burgers",
    resolution=32,
    n_samples=1000
)

# Access data
print(f"Train inputs shape: {navier_stokes_data.train['inputs'].shape}")
print(f"Train targets shape: {navier_stokes_data.train['targets'].shape}")
print(f"Equation type: {navier_stokes_data.metadata['equation']}")
```

### `qnet_no.utils`

#### Quantum Utility Functions

```python
# Quantum Fourier operations
from qnet_no.utils.quantum_fourier import quantum_fourier_modes

def quantum_fourier_modes(x: jnp.ndarray, modes: int, network: PhotonicNetwork, 
                         schmidt_rank: int, inverse: bool = False) -> jnp.ndarray
```

```python
# Quantum encoding
from qnet_no.utils.quantum_encoding import quantum_feature_map

def quantum_feature_map(data: jnp.ndarray, network: PhotonicNetwork, 
                       schmidt_rank: int, encoding_type: str = "amplitude") -> jnp.ndarray
```

```python
# Tensor operations
from qnet_no.utils.tensor_ops import distributed_dot_product, tensor_product_einsum

def distributed_dot_product(a: jnp.ndarray, b: jnp.ndarray, 
                           network: PhotonicNetwork) -> jnp.ndarray

def tensor_product_einsum(equation: str, *operands, network: PhotonicNetwork) -> jnp.ndarray
```

#### Globalization

```python
from qnet_no.utils.globalization import (
    translate, set_global_locale, validate_region_deployment,
    SupportedLocale, SupportedRegion
)

# Set locale
set_global_locale(SupportedLocale.JA_JP)

# Translate text
translated = translate("quantum_advantage")  # Returns: "量子優位性"

# Validate deployment
config = {"encryption": {"at_rest": True}, "compliance": ["GDPR"]}
validation = validate_region_deployment(SupportedRegion.EU_WEST, config)
```

### `qnet_no.backends`

#### Quantum Backends

```python
from qnet_no.backends import SimulatorBackend, PhotonicBackend, NVCenterBackend

# Simulator backend
backend = SimulatorBackend(n_qubits=8)
backend.connect()
result = backend.execute_circuit(circuit, shots=1000)
backend.disconnect()

# Photonic backend (requires Xanadu credentials)
photonic = PhotonicBackend(device="X8", api_key="your_key")
if photonic.connect():
    result = photonic.execute_circuit(photonic_circuit)

# NV Center backend  
nv_backend = NVCenterBackend(endpoint="https://nv.example.com")
capabilities = nv_backend.get_backend_properties()
```

## 🔬 Research & Experimental APIs

### `research.experimental_framework`

```python
from research.experimental_framework import (
    ExperimentalFramework, ExperimentConfig, ExperimentType,
    create_comprehensive_study, run_quantum_advantage_certification_study
)

# Create experiment
config = ExperimentConfig(
    experiment_name="quantum_advantage_study",
    experiment_type=ExperimentType.SCHEDULING_OPTIMIZATION,
    n_trials=50,
    network_sizes=[4, 8, 16, 32]
)

framework = ExperimentalFramework(config)
results = framework.run_full_experimental_suite()

# Quick certification study
certification_results = run_quantum_advantage_certification_study(
    network_size=16,
    n_trials=100
)
```

### Baseline Implementations

```python
from research.experimental_framework import BaselineImplementations

baselines = BaselineImplementations()

# Classical scheduling baselines
tasks = [...]  # List of ComputationTask
network = PhotonicNetwork(nodes=4)

greedy_assignment, greedy_time = baselines.classical_greedy_scheduler(tasks, network)
sa_assignment, sa_time = baselines.classical_simulated_annealing(tasks, network, max_iter=1000)
random_assignment, random_time = baselines.quantum_random_baseline(tasks, network, n_samples=100)
```

## 📊 Configuration Classes

### `HybridSchedulingConfig`

```python
@dataclass
class HybridSchedulingConfig:
    qaoa_layers: int = 4
    optimization_steps: int = 100
    adaptive_schmidt_rank: bool = True
    multi_fidelity_optimization: bool = True
    quantum_advantage_certification: bool = True
    real_time_adaptation: bool = True
    classical_fallback_threshold: float = 0.95
```

### `RegionConfig` (Globalization)

```python
@dataclass 
class RegionConfig:
    region: SupportedRegion
    primary_locale: SupportedLocale
    supported_locales: List[SupportedLocale]
    compliance_requirements: List[str]
    currency: str
    timezone: str
    quantum_hardware_available: bool = False
    entanglement_distance_limit_km: float = 1000.0
```

## 📈 Performance Monitoring APIs

### Metrics Collection

```python
from qnet_no.utils.metrics import get_metrics_collector

collector = get_metrics_collector()

# Record quantum metrics
collector.record_quantum_metrics(
    circuit_fidelity=0.94,
    entanglement_quality=0.87,
    schmidt_rank=16
)

# Record training metrics
collector.record_training_metrics(
    epoch=50,
    loss=0.0045,
    learning_rate=1e-3,
    batch_size=32,
    throughput=1200
)

# Get performance report
report = collector.get_performance_report()
print(f"Average quantum advantage: {report['avg_quantum_advantage']:.2f}")
```

### Performance Optimization

```python
# Memory pool management
qfno.memory_pool.expand_pool(factor=2.0)
qfno.memory_pool.get_statistics()

# Computation caching
qfno.computation_cache.set_cache_size(max_size_gb=5.0)
qfno.computation_cache.cleanup_cache(target_size_gb=3.0)

# Auto-scaling
optimal_batch = qfno.auto_scale_batch_size(
    current_loss=0.045,
    memory_usage=3.2e9,  # bytes
    throughput=1200      # samples/sec
)
```

## 🚨 Error Handling

### Exception Classes

```python
from qnet_no.utils.error_handling import (
    OperatorError, TrainingError, NetworkError, QuantumError,
    ErrorSeverity, error_boundary
)

try:
    result = qfno.fit(train_data, network)
except TrainingError as e:
    print(f"Training failed: {e}")
    print(f"Severity: {e.severity}")
    print(f"Suggestions: {e.suggestions}")
except QuantumError as e:
    print(f"Quantum operation failed: {e}")
    if e.severity == ErrorSeverity.HIGH:
        # Handle critical quantum error
        network.reset_entanglement()
```

### Error Boundaries

```python
from qnet_no.utils.error_handling import error_boundary, monitor_resources

@error_boundary(TrainingError, ErrorSeverity.MEDIUM)
@monitor_resources("custom_training")
def custom_training_function(data, network):
    # Training logic here
    return results
```

## 🔧 Advanced Usage Patterns

### Custom Quantum Circuits

```python
from qnet_no.backends.base_backend import QuantumCircuit

# Define custom quantum circuit
custom_circuit = QuantumCircuit(
    gates=[
        {"gate": "h", "qubit": 0},
        {"gate": "cnot", "control": 0, "target": 1},
        {"gate": "ry", "qubit": 2, "angle": 0.5},
        {"gate": "measure", "qubit": [0, 1, 2]}
    ],
    n_qubits=3,
    measurements=[0, 1, 2]
)

# Execute on backend
backend = SimulatorBackend(n_qubits=3)
backend.connect()
result = backend.execute_circuit(custom_circuit, shots=1000)
print(f"Measurement results: {result.measurement_counts}")
```

### Custom Scheduling Algorithms

```python
from qnet_no.algorithms.hybrid_scheduling import MultiObjectiveQuantumOptimizer

class CustomQuantumOptimizer(MultiObjectiveQuantumOptimizer):
    def _construct_cost_hamiltonian(self, tasks, network):
        # Custom cost Hamiltonian construction
        cost_h = super()._construct_cost_hamiltonian(tasks, network)
        
        # Add custom terms
        for i, task in enumerate(tasks):
            if task.priority == TaskPriority.CRITICAL:
                cost_h[i, i] += 10.0  # High penalty for unassigned critical tasks
                
        return cost_h

# Use custom optimizer
config = HybridSchedulingConfig(qaoa_layers=6)
optimizer = CustomQuantumOptimizer(config)
```

### Multi-Network Operations

```python
# Create multiple networks for different regions
networks = {
    'us-east': PhotonicNetwork(nodes=8, topology='complete'),
    'eu-west': PhotonicNetwork(nodes=6, topology='ring'), 
    'asia': PhotonicNetwork(nodes=4, topology='star')
}

# Distribute computation across networks
results = {}
for region, network in networks.items():
    regional_tasks = filter_tasks_by_region(tasks, region)
    scheduler = create_hybrid_scheduler(network)
    results[region] = scheduler.schedule_tasks_hybrid(regional_tasks)

# Aggregate results
total_quantum_advantage = sum(r.quantum_advantage_score for r in results.values()) / len(results)
```

## 📝 Type Hints and Data Structures

### Core Data Types

```python
from typing import Dict, List, Optional, Tuple, Any, Union
import jax.numpy as jnp

# Common type aliases
NetworkTopology = Literal['complete', 'ring', 'star', 'grid', 'random']
QuantumBackendType = Literal['simulator', 'photonic', 'nv_center', 'superconducting']
TaskOperationType = Literal['fourier_transform', 'tensor_contraction', 'gate_sequence', 'measurement']

# Data container types
TrainingData = Dict[str, jnp.ndarray]  # {'inputs': ..., 'targets': ...}
NetworkStats = Dict[str, Union[int, float, bool]]
PerformanceMetrics = Dict[str, float]
ValidationResult = Dict[str, Union[bool, List[str], float]]
```

### Result Objects

```python
@dataclass
class SchedulingResult:
    task_assignments: Dict[str, int]
    execution_order: List[str] 
    estimated_completion_time: float
    resource_utilization: Dict[int, float]
    entanglement_usage: Dict[Tuple[int, int], float]
    quantum_advantage_score: float = 1.0

@dataclass
class DataContainer:
    train: TrainingData
    test: TrainingData
    metadata: Dict[str, Any]

@dataclass  
class CircuitResult:
    measurement_counts: Dict[str, int]
    state_vector: Optional[jnp.ndarray]
    fidelity: float
    execution_time: float
```

## 🔗 Integration Examples

### Flask/FastAPI Integration

```python
from flask import Flask, request, jsonify
import qnet_no as qno

app = Flask(__name__)

# Global network and scheduler
network = qno.networks.PhotonicNetwork(nodes=8)
scheduler = qno.algorithms.create_hybrid_scheduler(network)

@app.route('/api/schedule', methods=['POST'])
def schedule_tasks():
    task_data = request.json
    tasks = [qno.networks.ComputationTask(**task) for task in task_data['tasks']]
    
    result = scheduler.schedule_tasks_hybrid(tasks)
    
    return jsonify({
        'assignments': result.task_assignments,
        'quantum_advantage': result.quantum_advantage_score,
        'completion_time': result.estimated_completion_time
    })

@app.route('/api/train', methods=['POST']) 
def train_operator():
    config = request.json
    
    # Create operator
    qfno = qno.operators.QuantumFourierNeuralOperator(**config['operator_params'])
    
    # Load data
    data = getattr(qno.datasets, config['dataset'])(**config['data_params'])
    
    # Train
    results = qfno.fit(data.train, network, **config['training_params'])
    
    return jsonify({
        'final_loss': float(results['losses'][-1]),
        'training_time': results.get('training_time', 0),
        'convergence_epochs': len(results['losses'])
    })
```

### Jupyter Notebook Integration

```python
# Cell 1: Setup
%pip install qnet-no
import qnet_no as qno
from qnet_no.algorithms import create_hybrid_scheduler
from qnet_no.operators import QuantumFourierNeuralOperator

# Cell 2: Create and visualize network
network = qno.networks.PhotonicNetwork(nodes=6, topology='ring')
network.visualize()  # If visualization utils available

# Cell 3: Interactive training
qfno = QuantumFourierNeuralOperator(modes=16, schmidt_rank=8)
data = qno.datasets.load_heat_equation(resolution=32, n_samples=500)

# Enable progress tracking
from tqdm.notebook import tqdm
results = qfno.fit(data.train, network, epochs=50, progress_callback=tqdm)

# Cell 4: Results analysis  
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(results['losses'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(1, 2, 2)
predictions = qfno.predict(data.test, network)
plt.imshow(predictions[0, :, :, 0])
plt.title('Prediction Example')
plt.colorbar()

plt.tight_layout()
plt.show()
```

---

## 📞 Support & Resources

- **📚 Full Documentation**: [docs.qnet-no.ai](https://docs.qnet-no.ai)
- **🎓 Tutorials**: [tutorials.qnet-no.ai](https://tutorials.qnet-no.ai) 
- **💬 Community**: [discord.gg/qnet-no](https://discord.gg/qnet-no)
- **🐛 Issues**: [github.com/danieleschmidt/qnet-no/issues](https://github.com/danieleschmidt/qnet-no/issues)
- **📧 Contact**: [support@terragonlabs.ai](mailto:support@terragonlabs.ai)

For enterprise support and custom integrations, please contact [enterprise@terragonlabs.ai](mailto:enterprise@terragonlabs.ai).