"""Dataset loading and preprocessing for quantum neural operators."""

from .pde_datasets import (
    load_navier_stokes,
    load_heat_equation,
    load_wave_equation,
    load_burgers_equation,
    load_darcy_flow,
    load_maxwell_equations
)
from .synthetic_data import (
    generate_synthetic_pde_data,
    generate_operator_learning_data,
    create_benchmark_suite
)

__all__ = [
    "load_navier_stokes",
    "load_heat_equation", 
    "load_wave_equation",
    "load_burgers_equation",
    "load_darcy_flow",
    "load_maxwell_equations",
    "generate_synthetic_pde_data",
    "generate_operator_learning_data",
    "create_benchmark_suite",
]