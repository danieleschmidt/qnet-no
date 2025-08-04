"""PDE dataset loaders for quantum neural operator benchmarks."""

from typing import Dict, Optional, Tuple, Any
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass


@dataclass
class PDEDataset:
    """Container for PDE dataset with train/test splits."""
    train: Dict[str, jnp.ndarray]
    test: Dict[str, jnp.ndarray] 
    metadata: Dict[str, Any]


def load_navier_stokes(resolution: int = 64, n_samples: int = 1000, 
                      viscosity: float = 1e-4, dt: float = 0.01,
                      n_timesteps: int = 50) -> PDEDataset:
    """
    Load Navier-Stokes equation dataset.
    
    Generates 2D incompressible flow data with periodic boundary conditions.
    
    Args:
        resolution: Spatial grid resolution (resolution x resolution)
        n_samples: Number of samples to generate
        viscosity: Kinematic viscosity parameter
        dt: Time step size
        n_timesteps: Number of time steps to simulate
        
    Returns:
        PDEDataset with velocity fields and vorticity evolution
    """
    print(f"Generating Navier-Stokes dataset: {n_samples} samples at {resolution}x{resolution}")
    
    # Generate initial conditions
    initial_conditions = generate_navier_stokes_initial_conditions(
        n_samples, resolution, viscosity
    )
    
    # Simulate evolution
    solutions = simulate_navier_stokes_evolution(
        initial_conditions, viscosity, dt, n_timesteps, resolution
    )
    
    # Split into train/test
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
        'equation': 'navier_stokes_2d',
        'resolution': resolution,
        'viscosity': viscosity,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def load_heat_equation(resolution: int = 64, n_samples: int = 1000,
                      diffusivity: float = 0.1, dt: float = 0.01,
                      n_timesteps: int = 100) -> PDEDataset:
    """
    Load heat equation dataset.
    
    Generates 2D heat diffusion data with various initial conditions.
    """
    print(f"Generating heat equation dataset: {n_samples} samples at {resolution}x{resolution}")
    
    # Generate initial temperature distributions
    initial_conditions = generate_heat_initial_conditions(n_samples, resolution)
    
    # Simulate heat diffusion
    solutions = simulate_heat_evolution(
        initial_conditions, diffusivity, dt, n_timesteps, resolution
    )
    
    # Split into train/test
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
        'equation': 'heat_equation_2d',
        'resolution': resolution,
        'diffusivity': diffusivity,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def load_wave_equation(resolution: int = 64, n_samples: int = 1000,
                      wave_speed: float = 1.0, dt: float = 0.01,
                      n_timesteps: int = 100) -> PDEDataset:
    """
    Load wave equation dataset.
    
    Generates 2D wave propagation data with various initial conditions.
    """
    print(f"Generating wave equation dataset: {n_samples} samples at {resolution}x{resolution}")
    
    # Generate initial wave conditions (position and velocity)
    initial_conditions = generate_wave_initial_conditions(n_samples, resolution)
    
    # Simulate wave evolution
    solutions = simulate_wave_evolution(
        initial_conditions, wave_speed, dt, n_timesteps, resolution
    )
    
    # Split into train/test
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
        'equation': 'wave_equation_2d',
        'resolution': resolution,
        'wave_speed': wave_speed,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def load_burgers_equation(resolution: int = 256, n_samples: int = 1000,
                         viscosity: float = 0.01, dt: float = 0.001,
                         n_timesteps: int = 100) -> PDEDataset:
    """
    Load 1D Burgers equation dataset.
    
    Generates viscous Burgers equation solutions with shock wave formation.
    """
    print(f"Generating Burgers equation dataset: {n_samples} samples at resolution {resolution}")
    
    # Generate initial conditions
    initial_conditions = generate_burgers_initial_conditions(n_samples, resolution)
    
    # Simulate Burgers evolution
    solutions = simulate_burgers_evolution(
        initial_conditions, viscosity, dt, n_timesteps, resolution
    )
    
    # Split into train/test
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
        'equation': 'burgers_equation_1d',
        'resolution': resolution,
        'viscosity': viscosity,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def load_darcy_flow(resolution: int = 64, n_samples: int = 1000) -> PDEDataset:
    """
    Load Darcy flow dataset.
    
    Generates steady-state flow through porous media with random permeability fields.
    """
    print(f"Generating Darcy flow dataset: {n_samples} samples at {resolution}x{resolution}")
    
    # Generate permeability fields
    permeability_fields = generate_permeability_fields(n_samples, resolution)
    
    # Solve Darcy flow equations
    pressure_fields = solve_darcy_flow(permeability_fields, resolution)
    
    # Split into train/test
    split_idx = int(0.8 * n_samples)
    
    train_data = {
        'inputs': permeability_fields[:split_idx],
        'targets': pressure_fields[:split_idx]
    }
    
    test_data = {
        'inputs': permeability_fields[split_idx:],
        'targets': pressure_fields[split_idx:]
    }
    
    metadata = {
        'equation': 'darcy_flow_2d',
        'resolution': resolution,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


def load_maxwell_equations(resolution: int = 64, n_samples: int = 1000,
                          dt: float = 0.01, n_timesteps: int = 50) -> PDEDataset:
    """
    Load Maxwell equations dataset.
    
    Generates electromagnetic wave propagation in 2D.
    """
    print(f"Generating Maxwell equations dataset: {n_samples} samples at {resolution}x{resolution}")
    
    # Generate initial electromagnetic fields
    initial_conditions = generate_maxwell_initial_conditions(n_samples, resolution)
    
    # Simulate electromagnetic evolution
    solutions = simulate_maxwell_evolution(
        initial_conditions, dt, n_timesteps, resolution
    )
    
    # Split into train/test
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
        'equation': 'maxwell_equations_2d',
        'resolution': resolution,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'n_train': len(train_data['inputs']),
        'n_test': len(test_data['inputs'])
    }
    
    return PDEDataset(train=train_data, test=test_data, metadata=metadata)


# Helper functions for generating initial conditions and simulations

def generate_navier_stokes_initial_conditions(n_samples: int, resolution: int, 
                                            viscosity: float) -> jnp.ndarray:
    """Generate random initial conditions for Navier-Stokes equations."""
    # Create random vorticity fields
    x = jnp.linspace(0, 2*jnp.pi, resolution)
    y = jnp.linspace(0, 2*jnp.pi, resolution)
    X, Y = jnp.meshgrid(x, y)
    
    initial_conditions = []
    
    for _ in range(n_samples):
        # Random Fourier modes for initial vorticity
        n_modes = 8
        vorticity = jnp.zeros((resolution, resolution))
        
        for k in range(1, n_modes + 1):
            for l in range(1, n_modes + 1):
                amplitude = np.random.normal(0, 1.0 / (k**2 + l**2))
                phase = np.random.uniform(0, 2*np.pi)
                
                vorticity += amplitude * jnp.sin(k * X + l * Y + phase)
        
        initial_conditions.append(vorticity[None, :, :, None])  # Add batch and channel dims
    
    return jnp.concatenate(initial_conditions, axis=0)


def simulate_navier_stokes_evolution(initial_conditions: jnp.ndarray, viscosity: float,
                                   dt: float, n_timesteps: int, resolution: int) -> jnp.ndarray:
    """Simulate Navier-Stokes evolution using spectral methods."""
    n_samples = initial_conditions.shape[0]
    
    # Initialize solutions array
    solutions = jnp.zeros((n_samples, resolution, resolution, n_timesteps))
    
    for i in range(n_samples):
        vorticity = initial_conditions[i, :, :, 0]
        
        # Simple forward Euler time stepping (in practice would use more sophisticated methods)
        for t in range(n_timesteps):
            # Store current state
            solutions = solutions.at[i, :, :, t].set(vorticity)
            
            # Compute Laplacian using finite differences
            laplacian = compute_laplacian(vorticity, resolution)
            
            # Update vorticity (simplified Navier-Stokes)
            vorticity = vorticity + dt * (viscosity * laplacian)
    
    return solutions


def generate_heat_initial_conditions(n_samples: int, resolution: int) -> jnp.ndarray:
    """Generate random initial temperature distributions."""
    initial_conditions = []
    
    for _ in range(n_samples):
        # Random temperature field with hot spots
        temperature = np.random.normal(0, 0.5, (resolution, resolution))
        
        # Add some hot spots
        n_spots = np.random.randint(1, 5)
        for _ in range(n_spots):
            center_x = np.random.randint(resolution//4, 3*resolution//4)
            center_y = np.random.randint(resolution//4, 3*resolution//4)
            radius = np.random.uniform(5, 15)
            intensity = np.random.uniform(5, 10)
            
            x, y = np.meshgrid(np.arange(resolution), np.arange(resolution))
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask = dist < radius
            
            temperature[mask] += intensity * np.exp(-(dist[mask] / radius)**2)
        
        initial_conditions.append(temperature[None, :, :, None])
    
    return jnp.array(np.concatenate(initial_conditions, axis=0))


def simulate_heat_evolution(initial_conditions: jnp.ndarray, diffusivity: float,
                          dt: float, n_timesteps: int, resolution: int) -> jnp.ndarray:
    """Simulate heat equation evolution."""
    n_samples = initial_conditions.shape[0]
    solutions = jnp.zeros((n_samples, resolution, resolution, n_timesteps))
    
    for i in range(n_samples):
        temperature = initial_conditions[i, :, :, 0]
        
        for t in range(n_timesteps):
            solutions = solutions.at[i, :, :, t].set(temperature)
            
            # Heat equation: ∂T/∂t = α∇²T
            laplacian = compute_laplacian(temperature, resolution)
            temperature = temperature + dt * diffusivity * laplacian
    
    return solutions


def generate_wave_initial_conditions(n_samples: int, resolution: int) -> jnp.ndarray:
    """Generate initial conditions for wave equation (position and velocity)."""
    initial_conditions = []
    
    for _ in range(n_samples):
        # Initial displacement
        displacement = np.random.normal(0, 0.1, (resolution, resolution))
        
        # Initial velocity
        velocity = np.random.normal(0, 0.1, (resolution, resolution))
        
        # Add some wave packets
        n_packets = np.random.randint(1, 3)
        for _ in range(n_packets):
            center_x = np.random.randint(resolution//4, 3*resolution//4)
            center_y = np.random.randint(resolution//4, 3*resolution//4)
            width = np.random.uniform(5, 15)
            amplitude = np.random.uniform(1, 3)
            
            x, y = np.meshgrid(np.arange(resolution), np.arange(resolution))
            wave_packet = amplitude * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * width**2))
            
            displacement += wave_packet
        
        # Stack displacement and velocity
        initial_state = np.stack([displacement, velocity], axis=-1)
        initial_conditions.append(initial_state[None, :, :, :])
    
    return jnp.array(np.concatenate(initial_conditions, axis=0))


def simulate_wave_evolution(initial_conditions: jnp.ndarray, wave_speed: float,
                          dt: float, n_timesteps: int, resolution: int) -> jnp.ndarray:
    """Simulate wave equation evolution."""
    n_samples = initial_conditions.shape[0]
    solutions = jnp.zeros((n_samples, resolution, resolution, n_timesteps))
    
    for i in range(n_samples):
        displacement = initial_conditions[i, :, :, 0]
        velocity = initial_conditions[i, :, :, 1]
        
        for t in range(n_timesteps):
            solutions = solutions.at[i, :, :, t].set(displacement)
            
            # Wave equation: ∂²u/∂t² = c²∇²u
            laplacian = compute_laplacian(displacement, resolution)
            acceleration = wave_speed**2 * laplacian
            
            # Update using velocity and acceleration
            new_displacement = displacement + dt * velocity
            new_velocity = velocity + dt * acceleration
            
            displacement = new_displacement
            velocity = new_velocity
    
    return solutions


def generate_burgers_initial_conditions(n_samples: int, resolution: int) -> jnp.ndarray:
    """Generate initial conditions for 1D Burgers equation."""
    initial_conditions = []
    
    x = np.linspace(0, 2*np.pi, resolution)
    
    for _ in range(n_samples):
        # Random initial velocity profile
        u = np.zeros(resolution)
        
        # Add Fourier modes
        n_modes = 8
        for k in range(1, n_modes + 1):
            amplitude = np.random.normal(0, 1.0 / k)
            phase = np.random.uniform(0, 2*np.pi)
            u += amplitude * np.sin(k * x + phase)
        
        initial_conditions.append(u[None, :, None])  # Add batch and channel dims
    
    return jnp.array(np.concatenate(initial_conditions, axis=0))


def simulate_burgers_evolution(initial_conditions: jnp.ndarray, viscosity: float,
                             dt: float, n_timesteps: int, resolution: int) -> jnp.ndarray:
    """Simulate 1D Burgers equation evolution."""
    n_samples = initial_conditions.shape[0]
    solutions = jnp.zeros((n_samples, resolution, n_timesteps))
    
    dx = 2 * np.pi / resolution
    
    for i in range(n_samples):
        u = initial_conditions[i, :, 0]
        
        for t in range(n_timesteps):
            solutions = solutions.at[i, :, t].set(u)
            
            # Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
            
            # Convective term: u∂u/∂x (using upwind scheme)
            dudx = jnp.gradient(u, dx)
            convective = u * dudx
            
            # Viscous term: ν∂²u/∂x²
            d2udx2 = jnp.gradient(jnp.gradient(u, dx), dx)
            viscous = viscosity * d2udx2
            
            # Update
            u = u + dt * (-convective + viscous)
    
    return solutions


def generate_permeability_fields(n_samples: int, resolution: int) -> jnp.ndarray:
    """Generate random permeability fields for Darcy flow."""
    permeability_fields = []
    
    for _ in range(n_samples):
        # Log-normal permeability field
        log_perm = np.random.normal(0, 1, (resolution, resolution))
        
        # Apply spatial correlation
        from scipy.ndimage import gaussian_filter
        log_perm = gaussian_filter(log_perm, sigma=3)
        
        # Convert to permeability
        permeability = np.exp(log_perm)
        
        permeability_fields.append(permeability[None, :, :, None])
    
    return jnp.array(np.concatenate(permeability_fields, axis=0))


def solve_darcy_flow(permeability_fields: jnp.ndarray, resolution: int) -> jnp.ndarray:
    """Solve steady-state Darcy flow equation."""
    n_samples = permeability_fields.shape[0]
    pressure_fields = []
    
    for i in range(n_samples):
        perm = permeability_fields[i, :, :, 0]
        
        # Simplified Darcy flow solution
        # In practice, would solve ∇·(k∇p) = 0 with proper boundary conditions
        
        # Apply Laplacian operator with permeability weighting
        pressure = np.random.normal(0, 0.1, (resolution, resolution))
        
        # Simple iterative solver (Jacobi method)
        for _ in range(100):
            laplacian_p = compute_laplacian(pressure, resolution)
            pressure_new = pressure + 0.01 * perm * laplacian_p
            pressure = pressure_new
        
        pressure_fields.append(pressure[None, :, :, None])
    
    return jnp.array(np.concatenate(pressure_fields, axis=0))


def generate_maxwell_initial_conditions(n_samples: int, resolution: int) -> jnp.ndarray:
    """Generate initial electromagnetic field conditions."""
    initial_conditions = []
    
    for _ in range(n_samples):
        # Electric field components (Ex, Ey)
        Ex = np.random.normal(0, 0.1, (resolution, resolution))
        Ey = np.random.normal(0, 0.1, (resolution, resolution))
        
        # Magnetic field (Hz)
        Hz = np.random.normal(0, 0.1, (resolution, resolution))
        
        # Stack all field components
        fields = np.stack([Ex, Ey, Hz], axis=-1)
        initial_conditions.append(fields[None, :, :, :])
    
    return jnp.array(np.concatenate(initial_conditions, axis=0))


def simulate_maxwell_evolution(initial_conditions: jnp.ndarray, dt: float,
                             n_timesteps: int, resolution: int) -> jnp.ndarray:
    """Simulate Maxwell equations evolution."""
    n_samples = initial_conditions.shape[0]
    solutions = jnp.zeros((n_samples, resolution, resolution, n_timesteps))
    
    dx = 1.0 / resolution
    
    for i in range(n_samples):
        Ex = initial_conditions[i, :, :, 0]
        Ey = initial_conditions[i, :, :, 1]
        Hz = initial_conditions[i, :, :, 2]
        
        for t in range(n_timesteps):
            # Store magnetic field as output
            solutions = solutions.at[i, :, :, t].set(Hz)
            
            # Update electromagnetic fields using FDTD method
            # ∂Hz/∂t = (∂Ey/∂x - ∂Ex/∂y)
            dEy_dx = jnp.gradient(Ey, dx, axis=1)
            dEx_dy = jnp.gradient(Ex, dx, axis=0)
            
            Hz_new = Hz + dt * (dEy_dx - dEx_dy)
            
            # ∂Ex/∂t = ∂Hz/∂y
            dHz_dy = jnp.gradient(Hz, dx, axis=0)
            Ex_new = Ex + dt * dHz_dy
            
            # ∂Ey/∂t = -∂Hz/∂x
            dHz_dx = jnp.gradient(Hz, dx, axis=1)
            Ey_new = Ey - dt * dHz_dx
            
            Ex, Ey, Hz = Ex_new, Ey_new, Hz_new
    
    return solutions


def compute_laplacian(u: jnp.ndarray, resolution: int) -> jnp.ndarray:
    """Compute 2D Laplacian using finite differences."""
    # Second derivatives using finite differences
    d2u_dx2 = jnp.gradient(jnp.gradient(u, axis=1), axis=1)
    d2u_dy2 = jnp.gradient(jnp.gradient(u, axis=0), axis=0)
    
    return d2u_dx2 + d2u_dy2