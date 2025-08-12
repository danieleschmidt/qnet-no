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

from .quantum_meta_learning import (
    QuantumMetaLearner,
    QuantumAlgorithmGenome,
    MetaLearningTask,
    QuantumCircuitGenerator,
    QuantumAlgorithmEvaluator,
    QuantumEvolutionaryOptimizer
)

from .quantum_federated_learning import (
    QuantumFederatedTrainer,
    QuantumFederatedClient,
    QuantumFederatedRound,
    QuantumHomomorphicEncryption,
    QuantumDifferentialPrivacy,
    QuantumSecureAggregation
)

from .realtime_quantum_advantage import (
    RealTimeQuantumAdvantageMonitor,
    QuantumAdvantageSnapshot,
    AdvantageAlertConfig,
    QuantumAdvantagePredictor,
    StreamingStatisticalTester,
    create_default_monitor
)

__all__ = [
    # Hybrid Scheduling
    'HybridQuantumClassicalScheduler',
    'HybridSchedulingConfig', 
    'AdaptiveSchmidtRankOptimizer',
    'MultiObjectiveQuantumOptimizer',
    'QuantumSchedulingDevice',
    'create_hybrid_scheduler',
    'benchmark_quantum_advantage',
    
    # Quantum Meta-Learning
    'QuantumMetaLearner',
    'QuantumAlgorithmGenome',
    'MetaLearningTask',
    'QuantumCircuitGenerator',
    'QuantumAlgorithmEvaluator',
    'QuantumEvolutionaryOptimizer',
    
    # Quantum Federated Learning
    'QuantumFederatedTrainer',
    'QuantumFederatedClient',
    'QuantumFederatedRound',
    'QuantumHomomorphicEncryption',
    'QuantumDifferentialPrivacy',
    'QuantumSecureAggregation',
    
    # Real-Time Quantum Advantage
    'RealTimeQuantumAdvantageMonitor',
    'QuantumAdvantageSnapshot',
    'AdvantageAlertConfig',
    'QuantumAdvantagePredictor',
    'StreamingStatisticalTester',
    'create_default_monitor'
]

# Version info for research tracking
__version__ = '1.0.0'
__research_contributions__ = [
    'First hybrid quantum-classical scheduling for distributed quantum neural operators',
    'Novel adaptive Schmidt rank optimization algorithm', 
    'Multi-objective quantum optimization with advantage certification',
    'Real-time performance adaptation and monitoring system',
    'World-first quantum meta-learning framework for autonomous algorithm discovery',
    'Comprehensive quantum federated learning with privacy-preserving protocols',
    'Real-time quantum advantage monitoring and optimization system',
    'Quantum transformer neural operators for distributed quantum networks'
]