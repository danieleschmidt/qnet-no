"""Synthetic data generation for quantum neural operator testing and benchmarking."""

from typing import Dict, List, Optional, Tuple, Any, Callable
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from .pde_datasets import PDEDataset


@dataclass
class BenchmarkSuite:
    """Collection of benchmark datasets for comprehensive evaluation."""
    datasets: Dict[str, PDEDataset]
    metadata: Dict[str, Any]


def generate_synthetic_pde_data(equation_type: str, resolution: int = 64,
                               n_samples: int = 1000, **kwargs) -> PDEDataset:
    """
    Generate synthetic PDE data for testing quantum neural operators.
    
    Args:
        equation_type: Type of PDE ('linear', 'nonlinear', 'hyperbolic', 'parabolic')
        resolution: Spatial resolution
        n_samples: Number of samples to generate
        **kwargs: Additional parameters specific to equation type
        
    Returns:
        PDEDataset with synthetic solutions
    """
    if equation_type == "linear":
        return generate_linear_pde_data(resolution, n_samples, **kwargs)
    elif equation_type == "nonlinear":
        return generate_nonlinear_pde_data(resolution, n_samples, **kwargs)
    elif equation_type == "hyperbolic":
        return generate_hyperbolic_pde_data(resolution, n_samples, **kwargs)
    elif equation_type == "parabolic":
        return generate_parabolic_pde_data(resolution, n_samples, **kwargs)
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")


def generate_operator_learning_data(operator_type: str, input_dim: int = 100,
                                  output_dim: int = 100, n_samples: int = 1000,
                                  **kwargs) -> Dict[str, jnp.ndarray]:
    """
    Generate data for operator learning tasks.
    
    Creates function-to-function mapping data suitable for DeepONet training.
    
    Args:
        operator_type: Type of operator ('integral', 'differential', 'nonlocal')
        input_dim: Dimension of input function space
        output_dim: Dimension of output function space  
        n_samples: Number of function pairs to generate
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with 'u' (input functions), 'y' (query points), 's' (outputs)
    """
    if operator_type == "integral":
        return generate_integral_operator_data(input_dim, output_dim, n_samples, **kwargs)
    elif operator_type == "differential":
        return generate_differential_operator_data(input_dim, output_dim, n_samples, **kwargs)
    elif operator_type == "nonlocal":
        return generate_nonlocal_operator_data(input_dim, output_dim, n_samples, **kwargs)
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def create_benchmark_suite(resolution: int = 64, n_samples_per_dataset: int = 500) -> BenchmarkSuite:
    """
    Create comprehensive benchmark suite for quantum neural operators.
    
    Generates multiple datasets with varying complexity and characteristics
    to thoroughly evaluate quantum advantage and model performance.
    """
    datasets = {}
    
    # Linear PDEs - baseline performance
    print("Generating linear PDE benchmarks...")
    datasets['linear_diffusion'] = generate_linear_pde_data(
        resolution, n_samples_per_dataset, pde_type='diffusion'
    )
    datasets['linear_advection'] = generate_linear_pde_data(
        resolution, n_samples_per_dataset, pde_type='advection'
    )
    
    # Nonlinear PDEs - test quantum advantage
    print("Generating nonlinear PDE benchmarks...")
    datasets['nonlinear_reaction'] = generate_nonlinear_pde_data(
        resolution, n_samples_per_dataset, nonlinearity='reaction'
    )
    datasets['nonlinear_cubic'] = generate_nonlinear_pde_data(
        resolution, n_samples_per_dataset, nonlinearity='cubic'
    )
    
    # Multi-scale problems - challenge for classical methods
    print("Generating multi-scale benchmarks...")
    datasets['multiscale_periodic'] = generate_multiscale_data(
        resolution, n_samples_per_dataset, scale_type='periodic'
    )
    datasets['multiscale_random'] = generate_multiscale_data(
        resolution, n_samples_per_dataset, scale_type='random'
    )
    
    # High-dimensional problems - quantum advantage domain
    print("Generating high-dimensional benchmarks...")
    datasets['highdim_tensor'] = generate_high_dimensional_data(
        resolution, n_samples_per_dataset, dimensions=3
    )
    
    # Operator learning tasks
    print("Generating operator learning benchmarks...")
    operator_data = generate_operator_learning_suite(n_samples_per_dataset)
    datasets.update(operator_data)
    
    metadata = {
        'suite_type': 'comprehensive_benchmark',
        'resolution': resolution,
        'n_datasets': len(datasets),
        'total_samples': len(datasets) * n_samples_per_dataset,
        'complexity_levels': ['linear', 'nonlinear', 'multiscale', 'highdim'],
        'quantum_advantage_expected': ['nonlinear', 'multiscale', 'highdim']
    }
    
    return BenchmarkSuite(datasets=datasets, metadata=metadata)


def generate_linear_pde_data(resolution: int, n_samples: int, 
                           pde_type: str = 'diffusion', **kwargs) -> PDEDataset:
    """Generate linear PDE data for baseline testing."""
    print(f"Generating linear {pde_type} PDE data...")
    
    if pde_type == 'diffusion':
        # Linear diffusion: ∂u/∂t = α∇²u
        alpha = kwargs.get('diffusion_coeff', 0.1)
        return generate_linear_diffusion_data(resolution, n_samples, alpha)
        
    elif pde_type == 'advection':
        # Linear advection: ∂u/∂t + c·∇u = 0
        velocity = kwargs.get('velocity', [1.0, 0.5])
        return generate_linear_advection_data(resolution, n_samples, velocity)
        
    else:
        raise ValueError(f"Unknown linear PDE type: {pde_type}")


def generate_nonlinear_pde_data(resolution: int, n_samples: int,
                               nonlinearity: str = 'reaction', **kwargs) -> PDEDataset:
    """Generate nonlinear PDE data to test quantum advantage."""
    print(f"Generating nonlinear {nonlinearity} PDE data...")
    
    if nonlinearity == 'reaction':
        # Reaction-diffusion: ∂u/∂t = D∇²u + f(u)
        return generate_reaction_diffusion_data(resolution, n_samples, **kwargs)
        
    elif nonlinearity == 'cubic':
        # Cubic nonlinearity: ∂u/∂t = ∇²u + u³
        return generate_cubic_nonlinear_data(resolution, n_samples, **kwargs)
        
    else:
        raise ValueError(f"Unknown nonlinearity type: {nonlinearity}")


def generate_linear_diffusion_data(resolution: int, n_samples: int, 
                                 alpha: float) -> PDEDataset:
    """Generate linear diffusion equation data."""
    # Generate initial conditions
    initial_conditions = []
    solutions = []
    
    x = jnp.linspace(0, 2*jnp.pi, resolution)
    y = jnp.linspace(0, 2*jnp.pi, resolution)
    X, Y = jnp.meshgrid(x, y)
    
    dt = 0.01
    n_timesteps = 50
    
    for _ in range(n_samples):
        # Random initial condition with multiple Gaussian peaks
        u0 = jnp.zeros((resolution, resolution))
        
        n_peaks = np.random.randint(2, 6)
        for _ in range(n_peaks):
            cx = np.random.uniform(0, 2*jnp.pi)
            cy = np.random.uniform(0, 2*jnp.pi)
            sigma = np.random.uniform(0.3, 0.8)
            amplitude = np.random.uniform(0.5, 2.0)
            
            u0 += amplitude * jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # Simulate diffusion
        u = u0
        final_state = None
        
        for t in range(n_timesteps):
            laplacian = compute_2d_laplacian(u, resolution)
            u = u + dt * alpha * laplacian
            
            if t == n_timesteps - 1:
                final_state = u
        
        initial_conditions.append(u0[None, :, :, None])
        solutions.append(final_state[None, :, :, None])
    
    initial_conditions = jnp.concatenate(initial_conditions, axis=0)
    solutions = jnp.concatenate(solutions, axis=0)
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'inputs': initial_conditions[:split_idx],
        'targets': solutions[:split_idx]
    }
    
    test_data = {
        'inputs': initial_conditions[split_idx:],
        'targets': solutions[split_idx:]
    }
    
    metadata = {
        'equation': 'linear_diffusion',
        'diffusion_coeff': alpha,
        'resolution': resolution,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def generate_linear_advection_data(resolution: int, n_samples: int,
                                 velocity: List[float]) -> PDEDataset:
    """Generate linear advection equation data."""
    initial_conditions = []
    solutions = []
    
    x = jnp.linspace(0, 2*jnp.pi, resolution)
    y = jnp.linspace(0, 2*jnp.pi, resolution)
    X, Y = jnp.meshgrid(x, y)
    
    dt = 0.005
    n_timesteps = 100
    dx = 2*jnp.pi / resolution
    
    vx, vy = velocity
    
    for _ in range(n_samples):
        # Create advected pattern
        u0 = jnp.zeros((resolution, resolution))
        
        # Add traveling wave pattern
        kx = np.random.uniform(1, 3)
        ky = np.random.uniform(1, 3)
        phase = np.random.uniform(0, 2*jnp.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        
        u0 = amplitude * jnp.sin(kx * X + ky * Y + phase)
        
        # Add Gaussian blob
        cx = np.random.uniform(jnp.pi/2, 3*jnp.pi/2)
        cy = np.random.uniform(jnp.pi/2, 3*jnp.pi/2)
        sigma = np.random.uniform(0.3, 0.6)
        blob_amp = np.random.uniform(0.5, 1.0)
        
        u0 += blob_amp * jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # Advect the pattern
        u = u0
        
        for t in range(n_timesteps):
            # Upwind finite difference scheme
            dudx = jnp.gradient(u, dx, axis=1)
            dudy = jnp.gradient(u, dx, axis=0)
            
            u = u - dt * (vx * dudx + vy * dudy)
        
        initial_conditions.append(u0[None, :, :, None])
        solutions.append(u[None, :, :, None])
    
    initial_conditions = jnp.concatenate(initial_conditions, axis=0)
    solutions = jnp.concatenate(solutions, axis=0)
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'inputs': initial_conditions[:split_idx],
        'targets': solutions[:split_idx]
    }
    
    test_data = {
        'inputs': initial_conditions[split_idx:],
        'targets': solutions[split_idx:]
    }
    
    metadata = {
        'equation': 'linear_advection',
        'velocity': velocity,
        'resolution': resolution,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def generate_reaction_diffusion_data(resolution: int, n_samples: int, **kwargs) -> PDEDataset:
    """Generate reaction-diffusion PDE data with nonlinear reactions."""
    D = kwargs.get('diffusion_coeff', 0.1)
    reaction_rate = kwargs.get('reaction_rate', 1.0)
    
    initial_conditions = []
    solutions = []
    
    x = jnp.linspace(0, 2*jnp.pi, resolution)
    y = jnp.linspace(0, 2*jnp.pi, resolution)
    X, Y = jnp.meshgrid(x, y)
    
    dt = 0.001
    n_timesteps = 100
    
    for _ in range(n_samples):
        # Random initial condition
        u0 = np.random.uniform(0.1, 0.9, (resolution, resolution))
        
        # Add some structure
        n_centers = np.random.randint(2, 5)
        for _ in range(n_centers):
            cx = np.random.uniform(0, 2*jnp.pi)
            cy = np.random.uniform(0, 2*jnp.pi)
            sigma = np.random.uniform(0.5, 1.0)
            amplitude = np.random.uniform(0.2, 0.5)
            
            u0 += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        u0 = jnp.clip(u0, 0, 1)  # Keep in valid range
        
        # Simulate reaction-diffusion
        u = u0
        
        for t in range(n_timesteps):
            laplacian = compute_2d_laplacian(u, resolution)
            
            # Nonlinear reaction term: u(1-u) (logistic growth)
            reaction = reaction_rate * u * (1 - u)
            
            u = u + dt * (D * laplacian + reaction)
            u = jnp.clip(u, 0, 1)  # Keep bounded
        
        initial_conditions.append(u0[None, :, :, None])
        solutions.append(u[None, :, :, None])
    
    initial_conditions = jnp.array(initial_conditions)
    solutions = jnp.array(solutions)
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'inputs': initial_conditions[:split_idx],
        'targets': solutions[:split_idx]
    }
    
    test_data = {
        'inputs': initial_conditions[split_idx:],
        'targets': solutions[split_idx:]
    }
    
    metadata = {
        'equation': 'reaction_diffusion',
        'diffusion_coeff': D,
        'reaction_rate': reaction_rate,
        'resolution': resolution,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def generate_multiscale_data(resolution: int, n_samples: int, 
                           scale_type: str = 'periodic') -> PDEDataset:
    """Generate multi-scale PDE data to challenge classical methods."""
    print(f"Generating multi-scale {scale_type} data...")
    
    initial_conditions = []
    solutions = []
    
    x = jnp.linspace(0, 2*jnp.pi, resolution)
    y = jnp.linspace(0, 2*jnp.pi, resolution)
    X, Y = jnp.meshgrid(x, y)
    
    for _ in range(n_samples):
        u0 = jnp.zeros((resolution, resolution))
        
        if scale_type == 'periodic':
            # Multiple scales with different frequencies
            scales = [1, 2, 4, 8, 16]
            for k in scales:
                amplitude = np.random.uniform(0.1, 0.5) / k  # Higher freq = lower amplitude
                phase_x = np.random.uniform(0, 2*jnp.pi)
                phase_y = np.random.uniform(0, 2*jnp.pi)
                
                u0 += amplitude * jnp.sin(k * X + phase_x) * jnp.sin(k * Y + phase_y)
        
        elif scale_type == 'random':
            # Random multi-scale structure
            n_scales = 5
            for _ in range(n_scales):
                # Random frequency
                kx = np.random.uniform(1, 16)
                ky = np.random.uniform(1, 16)
                
                # Random center and width
                cx = np.random.uniform(0, 2*jnp.pi)
                cy = np.random.uniform(0, 2*jnp.pi)
                sigma = np.random.uniform(0.1, 1.0)
                amplitude = np.random.uniform(0.1, 0.8)
                
                # Localized oscillation
                envelope = jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                oscillation = jnp.sin(kx * X + ky * Y)
                
                u0 += amplitude * envelope * oscillation
        
        # Apply some evolution (simplified)
        u = u0
        dt = 0.01
        diffusivity = 0.05
        
        for t in range(20):
            laplacian = compute_2d_laplacian(u, resolution)
            u = u + dt * diffusivity * laplacian
        
        initial_conditions.append(u0[None, :, :, None])
        solutions.append(u[None, :, :, None])
    
    initial_conditions = jnp.array(initial_conditions)
    solutions = jnp.array(solutions)
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'inputs': initial_conditions[:split_idx],
        'targets': solutions[:split_idx]
    }
    
    test_data = {
        'inputs': initial_conditions[split_idx:],
        'targets': solutions[split_idx:]
    }
    
    metadata = {
        'equation': f'multiscale_{scale_type}',
        'resolution': resolution,
        'scale_type': scale_type,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def generate_high_dimensional_data(resolution: int, n_samples: int,
                                 dimensions: int = 3) -> PDEDataset:
    """Generate high-dimensional PDE data for quantum advantage testing."""
    print(f"Generating {dimensions}D high-dimensional data...")
    
    if dimensions == 3:
        return generate_3d_diffusion_data(resolution, n_samples)
    else:
        raise NotImplementedError(f"{dimensions}D data generation not implemented")


def generate_3d_diffusion_data(resolution: int, n_samples: int) -> PDEDataset:
    """Generate 3D diffusion equation data."""
    # Reduce resolution for 3D to manage memory
    res_3d = min(resolution, 32)
    
    initial_conditions = []
    solutions = []
    
    x = jnp.linspace(0, 2*jnp.pi, res_3d)
    y = jnp.linspace(0, 2*jnp.pi, res_3d)
    z = jnp.linspace(0, 2*jnp.pi, res_3d)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    dt = 0.001
    n_timesteps = 20
    diffusivity = 0.1
    
    for _ in range(n_samples):
        # 3D initial condition
        u0 = jnp.zeros((res_3d, res_3d, res_3d))
        
        # Add 3D Gaussian blobs
        n_blobs = np.random.randint(2, 4)
        for _ in range(n_blobs):
            cx = np.random.uniform(jnp.pi/2, 3*jnp.pi/2)
            cy = np.random.uniform(jnp.pi/2, 3*jnp.pi/2)
            cz = np.random.uniform(jnp.pi/2, 3*jnp.pi/2)
            sigma = np.random.uniform(0.3, 0.8)
            amplitude = np.random.uniform(0.5, 1.5)
            
            u0 += amplitude * jnp.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2))
        
        # Simulate 3D diffusion
        u = u0
        
        for t in range(n_timesteps):
            laplacian_3d = compute_3d_laplacian(u, res_3d)
            u = u + dt * diffusivity * laplacian_3d
        
        initial_conditions.append(u0[None, :, :, :, None])
        solutions.append(u[None, :, :, :, None])
    
    initial_conditions = jnp.array(initial_conditions)
    solutions = jnp.array(solutions)
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'inputs': initial_conditions[:split_idx],
        'targets': solutions[:split_idx]
    }
    
    test_data = {
        'inputs': initial_conditions[split_idx:],
        'targets': solutions[split_idx:]
    }
    
    metadata = {
        'equation': '3d_diffusion',
        'resolution': res_3d,
        'dimensions': 3,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def generate_integral_operator_data(input_dim: int, output_dim: int, 
                                  n_samples: int, **kwargs) -> Dict[str, jnp.ndarray]:
    """Generate data for integral operator learning."""
    kernel_type = kwargs.get('kernel_type', 'gaussian')
    
    # Input functions
    x_sensors = jnp.linspace(0, 1, input_dim)
    y_queries = jnp.linspace(0, 1, output_dim)
    
    u_functions = []
    outputs = []
    
    for _ in range(n_samples):
        # Generate random input function
        u = generate_random_function(x_sensors)
        
        # Apply integral operator
        if kernel_type == 'gaussian':
            # Gaussian kernel: ∫ exp(-|x-y|²/σ²) u(y) dy
            sigma = 0.1
            output = jnp.zeros(output_dim)
            
            for i, y_val in enumerate(y_queries):
                kernel_values = jnp.exp(-(x_sensors - y_val)**2 / (2 * sigma**2))
                output = output.at[i].set(jnp.trapz(kernel_values * u, x_sensors))
        
        u_functions.append(u)
        outputs.append(output)
    
    u_functions = jnp.array(u_functions)
    outputs = jnp.array(outputs)
    
    # Create query points (same for all samples)
    y_queries_batch = jnp.tile(y_queries[None, :, None], (n_samples, 1, 1))
    
    return {
        'u': u_functions,
        'y': y_queries_batch,
        's': outputs
    }


def generate_differential_operator_data(input_dim: int, output_dim: int,
                                      n_samples: int, **kwargs) -> Dict[str, jnp.ndarray]:
    """Generate data for differential operator learning."""
    operator_order = kwargs.get('order', 2)
    
    x_points = jnp.linspace(0, 1, input_dim)
    y_queries = jnp.linspace(0, 1, output_dim)
    
    u_functions = []
    outputs = []
    
    for _ in range(n_samples):
        # Generate smooth random function
        u = generate_smooth_function(x_points)
        
        # Apply differential operator
        if operator_order == 1:
            # First derivative
            du_dx = jnp.gradient(u, x_points[1] - x_points[0])
            output = jnp.interp(y_queries, x_points, du_dx)
            
        elif operator_order == 2:
            # Second derivative (Laplacian)
            d2u_dx2 = jnp.gradient(jnp.gradient(u, x_points[1] - x_points[0]), x_points[1] - x_points[0])
            output = jnp.interp(y_queries, x_points, d2u_dx2)
        
        u_functions.append(u)
        outputs.append(output)
    
    u_functions = jnp.array(u_functions)
    outputs = jnp.array(outputs)
    
    y_queries_batch = jnp.tile(y_queries[None, :, None], (n_samples, 1, 1))
    
    return {
        'u': u_functions,
        'y': y_queries_batch,
        's': outputs
    }


def generate_operator_learning_suite(n_samples: int) -> Dict[str, PDEDataset]:
    """Generate suite of operator learning benchmark datasets."""
    suite = {}
    
    # Integral operators
    integral_data = generate_integral_operator_data(100, 100, n_samples)
    suite['integral_operator'] = create_operator_dataset(integral_data, 'integral_operator')
    
    # Differential operators
    diff_data = generate_differential_operator_data(100, 100, n_samples, order=2)
    suite['differential_operator'] = create_operator_dataset(diff_data, 'differential_operator')
    
    return suite


def create_operator_dataset(data: Dict[str, jnp.ndarray], name: str) -> PDEDataset:
    """Convert operator data to PDEDataset format."""
    n_samples = len(data['u'])
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'u': data['u'][:split_idx],
        'y': data['y'][:split_idx],
        's': data['s'][:split_idx]
    }
    
    test_data = {
        'u': data['u'][split_idx:],
        'y': data['y'][split_idx:],
        's': data['s'][split_idx:]
    }
    
    metadata = {
        'dataset_type': 'operator_learning',
        'operator_name': name,
        'input_dim': data['u'].shape[1],
        'output_dim': data['s'].shape[1],
        'n_train': len(train_data['u']),
        'n_test': len(test_data['u'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


# Helper functions

def compute_2d_laplacian(u: jnp.ndarray, resolution: int) -> jnp.ndarray:
    """Compute 2D Laplacian using finite differences."""
    dx = 2 * jnp.pi / resolution
    
    # Second derivatives
    d2u_dx2 = jnp.gradient(jnp.gradient(u, dx, axis=1), dx, axis=1)
    d2u_dy2 = jnp.gradient(jnp.gradient(u, dx, axis=0), dx, axis=0)
    
    return d2u_dx2 + d2u_dy2


def compute_3d_laplacian(u: jnp.ndarray, resolution: int) -> jnp.ndarray:
    """Compute 3D Laplacian using finite differences."""
    dx = 2 * jnp.pi / resolution
    
    # Second derivatives in all three dimensions
    d2u_dx2 = jnp.gradient(jnp.gradient(u, dx, axis=2), dx, axis=2)
    d2u_dy2 = jnp.gradient(jnp.gradient(u, dx, axis=1), dx, axis=1)
    d2u_dz2 = jnp.gradient(jnp.gradient(u, dx, axis=0), dx, axis=0)
    
    return d2u_dx2 + d2u_dy2 + d2u_dz2


def generate_random_function(x_points: jnp.ndarray) -> jnp.ndarray:
    """Generate random function using Fourier series."""
    n_modes = 10
    f = jnp.zeros_like(x_points)
    
    for k in range(1, n_modes + 1):
        amplitude = np.random.normal(0, 1.0 / k)
        phase = np.random.uniform(0, 2*jnp.pi)
        f += amplitude * jnp.sin(2 * jnp.pi * k * x_points + phase)
    
    return f


def generate_smooth_function(x_points: jnp.ndarray) -> jnp.ndarray:
    """Generate smooth random function using low-frequency modes."""
    n_modes = 5
    f = jnp.zeros_like(x_points)
    
    for k in range(1, n_modes + 1):
        amplitude = np.random.normal(0, 1.0 / k**2)  # Smoother decay
        phase = np.random.uniform(0, 2*jnp.pi)
        f += amplitude * jnp.sin(2 * jnp.pi * k * x_points + phase)
    
    return f


def cubic_nonlinear_data(resolution: int, n_samples: int, **kwargs) -> PDEDataset:
    """Generate cubic nonlinear PDE data."""
    # Implementation for cubic nonlinearity: ∂u/∂t = ∇²u + u³
    pass  # Placeholder for additional nonlinear PDEs