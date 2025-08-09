"""
QNet-NO Advanced Algorithms Module

This module contains cutting-edge research algorithms for quantum-enhanced
distributed computing, including novel hybrid quantum-classical optimization
techniques and adaptive resource allocation strategies.

Key Research Contributions:
- Hybrid Quantum-Classical Scheduling Optimization
- Adaptive Schmidt Rank Optimization
- Multi-Objective Quantum Resource Allocation
- Entanglement-Aware Neural Architecture Search

Author: Terry - Terragon Labs
Date: 2025-08-09
"""

from .hybrid_scheduling import (
    HybridQuantumClassicalScheduler,
    HybridSchedulingConfig,
    AdaptiveSchmidtRankOptimizer,
    MultiObjectiveQuantumOptimizer,
    QuantumSchedulingDevice,
    create_hybrid_scheduler,
    benchmark_quantum_advantage
)

__all__ = [
    'HybridQuantumClassicalScheduler',
    'HybridSchedulingConfig', 
    'AdaptiveSchmidtRankOptimizer',
    'MultiObjectiveQuantumOptimizer',
    'QuantumSchedulingDevice',
    'create_hybrid_scheduler',
    'benchmark_quantum_advantage'
]

# Version info for research tracking
__version__ = '1.0.0'
__research_contributions__ = [
    'First hybrid quantum-classical scheduling for distributed quantum neural operators',
    'Novel adaptive Schmidt rank optimization algorithm', 
    'Multi-objective quantum optimization with advantage certification',
    'Real-time performance adaptation and monitoring system'
]