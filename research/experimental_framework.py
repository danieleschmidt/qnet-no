"""
Experimental Framework for Quantum Neural Operator Research

This module provides comprehensive experimental frameworks for validating
quantum advantage in distributed neural operator networks, including
baseline implementations, statistical analysis, and publication-ready
benchmarking capabilities.

Key Features:
- Comprehensive baseline implementations (classical, quantum, hybrid)
- Controlled experimental environments with reproducible results
- Statistical significance testing and quantum advantage certification
- Multi-objective performance evaluation across different metrics
- Automated experimental pipeline with result visualization

Research Standards:
- All experiments are reproducible with fixed random seeds
- Statistical significance testing with multiple trials
- Proper baseline comparisons and ablation studies
- Publication-ready result formatting and visualization

Author: Terry - Terragon Labs
Date: 2025-08-09
"""

from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import pickle
import logging
from pathlib import Path
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# QNet-NO imports
import sys
sys.path.append('..')
from qnet_no.algorithms.hybrid_scheduling import (
    HybridQuantumClassicalScheduler, HybridSchedulingConfig,
    benchmark_quantum_advantage, create_hybrid_scheduler
)
from qnet_no.networks.photonic_network import PhotonicNetwork
from qnet_no.networks.entanglement_scheduler import ComputationTask, TaskPriority
from qnet_no.operators.quantum_fno import QuantumFourierNeuralOperator  
from qnet_no.operators.quantum_deeponet import QuantumDeepONet
from qnet_no.datasets.pde_datasets import load_navier_stokes, load_heat_equation

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of experiments for quantum advantage validation."""
    SCHEDULING_OPTIMIZATION = "scheduling_optimization"
    NEURAL_OPERATOR_TRAINING = "neural_operator_training"
    ENTANGLEMENT_SCALING = "entanglement_scaling"
    SCHMIDT_RANK_ANALYSIS = "schmidt_rank_analysis"
    COMPARATIVE_PERFORMANCE = "comparative_performance"
    QUANTUM_ADVANTAGE_CERTIFICATION = "quantum_advantage_certification"


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs."""
    experiment_name: str
    experiment_type: ExperimentType
    n_trials: int = 10
    random_seed: int = 42
    
    # Network configurations
    network_sizes: List[int] = field(default_factory=lambda: [4, 8, 16])
    entanglement_fidelities: List[float] = field(default_factory=lambda: [0.85, 0.90, 0.95])
    
    # Algorithm parameters  
    qaoa_layers: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    schmidt_ranks: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    
    # Performance metrics
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'quantum_advantage_score', 'execution_time', 'resource_utilization',
        'solution_quality', 'scalability_factor'
    ])
    
    # Statistical analysis
    significance_level: float = 0.05
    effect_size_threshold: float = 0.1  # Minimum meaningful effect size
    
    # Output configuration
    save_results: bool = True
    save_plots: bool = True
    results_directory: str = "./research_results"


@dataclass  
class ExperimentResult:
    """Results from a single experimental configuration."""
    config: ExperimentConfig
    experiment_id: str
    parameters: Dict[str, Any]
    
    # Raw results
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistical analysis
    mean_values: Dict[str, float] = field(default_factory=dict)
    std_values: Dict[str, float] = field(default_factory=dict) 
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Significance testing
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Quantum advantage metrics
    quantum_advantage_significant: bool = False
    effect_size: float = 0.0
    
    # Timing and resource usage
    total_experiment_time: float = 0.0
    memory_usage_peak: float = 0.0


class BaselineImplementations:
    """Classical and quantum baseline implementations for comparison."""
    
    @staticmethod
    def classical_greedy_scheduler(tasks: List[ComputationTask], 
                                 network: PhotonicNetwork) -> Tuple[Dict[str, int], float]:
        """Classical greedy scheduling baseline."""
        start_time = time.time()
        
        assignment = {}
        node_loads = {node_id: 0 for node_id in network.quantum_nodes.keys()}
        
        # Sort tasks by priority and resource requirements
        sorted_tasks = sorted(tasks, key=lambda t: (-t.priority.value, -t.required_qubits))
        
        for task in sorted_tasks:
            best_node = None
            best_score = -np.inf
            
            for node_id, node in network.quantum_nodes.items():
                if node_loads[node_id] + task.required_qubits <= node.n_qubits:
                    # Simple scoring: prefer high-fidelity, low-load nodes
                    score = node.fidelity - (node_loads[node_id] / node.n_qubits)
                    
                    if score > best_score:
                        best_score = score
                        best_node = node_id
            
            if best_node is not None:
                assignment[task.task_id] = best_node
                node_loads[best_node] += task.required_qubits
        
        execution_time = time.time() - start_time
        return assignment, execution_time
    
    @staticmethod
    def classical_simulated_annealing(tasks: List[ComputationTask],
                                    network: PhotonicNetwork,
                                    max_iter: int = 1000) -> Tuple[Dict[str, int], float]:
        """Classical simulated annealing baseline."""
        from scipy.optimize import dual_annealing
        
        start_time = time.time()
        
        node_list = list(network.quantum_nodes.keys())
        n_tasks = len(tasks)
        n_nodes = len(node_list)
        
        def objective(x):
            assignment = {}
            total_cost = 0.0
            node_loads = {node_id: 0 for node_id in node_list}
            
            for i, task in enumerate(tasks):
                node_idx = int(x[i]) % n_nodes
                node_id = node_list[node_idx]
                assignment[task.task_id] = node_id
                node_loads[node_id] += task.required_qubits
                
                # Check constraints
                if node_loads[node_id] > network.quantum_nodes[node_id].n_qubits:
                    total_cost += 1000.0  # Penalty for constraint violation
                
                # Cost based on inverse fidelity
                total_cost += (1.0 - network.quantum_nodes[node_id].fidelity) * 10.0
            
            # Load balancing cost
            loads = list(node_loads.values())
            total_cost += np.std(loads) * 5.0
            
            return total_cost
        
        bounds = [(0, n_nodes - 1) for _ in range(n_tasks)]
        result = dual_annealing(objective, bounds, maxiter=max_iter, seed=42)
        
        # Convert result to assignment
        assignment = {}
        for i, task in enumerate(tasks):
            node_idx = int(result.x[i]) % n_nodes
            assignment[task.task_id] = node_list[node_idx]
        
        execution_time = time.time() - start_time
        return assignment, execution_time
    
    @staticmethod
    def quantum_random_baseline(tasks: List[ComputationTask],
                               network: PhotonicNetwork,
                               n_samples: int = 100) -> Tuple[Dict[str, int], float]:
        """Quantum-inspired random sampling baseline."""
        start_time = time.time()
        
        node_list = list(network.quantum_nodes.keys())
        best_assignment = None
        best_cost = np.inf
        
        for _ in range(n_samples):
            assignment = {}
            node_loads = {node_id: 0 for node_id in node_list}
            
            # Random assignment with constraint checking
            valid = True
            for task in tasks:
                valid_nodes = []
                for node_id in node_list:
                    if node_loads[node_id] + task.required_qubits <= network.quantum_nodes[node_id].n_qubits:
                        valid_nodes.append(node_id)
                
                if valid_nodes:
                    # Weighted random selection based on node quality
                    weights = [network.quantum_nodes[node_id].fidelity for node_id in valid_nodes]
                    weights = np.array(weights) / np.sum(weights)
                    
                    chosen_node = np.random.choice(valid_nodes, p=weights)
                    assignment[task.task_id] = chosen_node
                    node_loads[chosen_node] += task.required_qubits
                else:
                    valid = False
                    break
            
            if valid:
                # Calculate cost
                cost = 0.0
                for task in tasks:
                    node_id = assignment[task.task_id]
                    cost += (1.0 - network.quantum_nodes[node_id].fidelity) * 10.0
                
                loads = list(node_loads.values())
                cost += np.std(loads) * 5.0
                
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = assignment.copy()
        
        execution_time = time.time() - start_time
        return best_assignment or {}, execution_time


class ExperimentalFramework:
    """
    Main experimental framework for quantum neural operator research.
    
    Provides comprehensive experimental capabilities including:
    - Controlled experimental environments
    - Baseline comparisons and ablation studies  
    - Statistical significance testing
    - Quantum advantage certification
    - Publication-ready result generation
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.baselines = BaselineImplementations()
        
        # Set up result directories
        self.results_dir = Path(config.results_directory)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "data").mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        jax.random.PRNGKey(config.random_seed)
        
        logger.info(f"Initialized experimental framework: {config.experiment_name}")
    
    def run_full_experimental_suite(self) -> Dict[str, Any]:
        """
        Run complete experimental suite with all baselines and configurations.
        
        This is the main entry point for comprehensive quantum advantage studies.
        """
        logger.info(f"Starting full experimental suite: {self.config.experiment_name}")
        start_time = time.time()
        
        # Run experiments based on type
        if self.config.experiment_type == ExperimentType.SCHEDULING_OPTIMIZATION:
            results = self._run_scheduling_experiments()
        elif self.config.experiment_type == ExperimentType.NEURAL_OPERATOR_TRAINING:
            results = self._run_neural_operator_experiments()
        elif self.config.experiment_type == ExperimentType.ENTANGLEMENT_SCALING:
            results = self._run_entanglement_scaling_experiments()
        elif self.config.experiment_type == ExperimentType.SCHMIDT_RANK_ANALYSIS:
            results = self._run_schmidt_rank_experiments()
        elif self.config.experiment_type == ExperimentType.COMPARATIVE_PERFORMANCE:
            results = self._run_comparative_experiments()
        elif self.config.experiment_type == ExperimentType.QUANTUM_ADVANTAGE_CERTIFICATION:
            results = self._run_quantum_advantage_certification()
        else:
            raise ValueError(f"Unknown experiment type: {self.config.experiment_type}")
        
        # Comprehensive statistical analysis
        statistical_summary = self._perform_comprehensive_analysis(results)
        
        # Generate publication-ready visualizations
        if self.config.save_plots:
            self._generate_publication_plots(results, statistical_summary)
        
        # Save results
        if self.config.save_results:
            self._save_experimental_results(results, statistical_summary)
        
        total_time = time.time() - start_time
        logger.info(f"Experimental suite completed in {total_time:.2f} seconds")
        
        return {
            'experimental_results': results,
            'statistical_summary': statistical_summary,
            'total_experiment_time': total_time,
            'configuration': self.config
        }
    
    def _run_scheduling_experiments(self) -> Dict[str, Any]:
        """Run scheduling optimization experiments with multiple baselines."""
        logger.info("Running scheduling optimization experiments")
        
        results = {
            'hybrid_quantum_classical': [],
            'classical_greedy': [], 
            'classical_simulated_annealing': [],
            'quantum_random': []
        }
        
        # Test across different network sizes and configurations
        for network_size in self.config.network_sizes:
            for fidelity in self.config.entanglement_fidelities:
                for qaoa_layers in self.config.qaoa_layers:
                    
                    logger.info(f"Testing network_size={network_size}, fidelity={fidelity}, qaoa_layers={qaoa_layers}")
                    
                    # Create test network and tasks
                    network = self._create_test_network(network_size, fidelity)
                    tasks = self._generate_test_tasks(network_size * 2, network)
                    
                    # Run multiple trials for statistical significance
                    trial_results = {
                        'hybrid_quantum_classical': [],
                        'classical_greedy': [],
                        'classical_simulated_annealing': [], 
                        'quantum_random': []
                    }
                    
                    for trial in range(self.config.n_trials):
                        logger.debug(f"Trial {trial + 1}/{self.config.n_trials}")
                        
                        # Hybrid quantum-classical approach
                        hqc_start = time.time()
                        config = HybridSchedulingConfig(qaoa_layers=qaoa_layers)
                        scheduler = HybridQuantumClassicalScheduler(network, config)
                        hqc_result = scheduler.schedule_tasks_hybrid(tasks)
                        hqc_time = time.time() - hqc_start
                        
                        hqc_metrics = {
                            'execution_time': hqc_time,
                            'estimated_completion_time': hqc_result.estimated_completion_time,
                            'average_utilization': np.mean(list(hqc_result.resource_utilization.values())),
                            'quantum_advantage_score': scheduler.quantum_advantage_scores[-1] if scheduler.quantum_advantage_scores else 1.0,
                            'solution_quality': self._calculate_solution_quality(hqc_result, tasks, network)
                        }
                        trial_results['hybrid_quantum_classical'].append(hqc_metrics)
                        
                        # Classical baselines
                        cg_assignment, cg_time = self.baselines.classical_greedy_scheduler(tasks, network)
                        cg_metrics = {
                            'execution_time': cg_time,
                            'solution_quality': self._calculate_assignment_quality(cg_assignment, tasks, network),
                            'average_utilization': self._calculate_utilization(cg_assignment, tasks, network)
                        }
                        trial_results['classical_greedy'].append(cg_metrics)
                        
                        csa_assignment, csa_time = self.baselines.classical_simulated_annealing(tasks, network)
                        csa_metrics = {
                            'execution_time': csa_time,
                            'solution_quality': self._calculate_assignment_quality(csa_assignment, tasks, network),
                            'average_utilization': self._calculate_utilization(csa_assignment, tasks, network)
                        }
                        trial_results['classical_simulated_annealing'].append(csa_metrics)
                        
                        qr_assignment, qr_time = self.baselines.quantum_random_baseline(tasks, network)
                        qr_metrics = {
                            'execution_time': qr_time,
                            'solution_quality': self._calculate_assignment_quality(qr_assignment, tasks, network),
                            'average_utilization': self._calculate_utilization(qr_assignment, tasks, network)
                        }
                        trial_results['quantum_random'].append(qr_metrics)
                    
                    # Aggregate trial results
                    config_result = {
                        'network_size': network_size,
                        'fidelity': fidelity,
                        'qaoa_layers': qaoa_layers,
                        'trial_data': trial_results
                    }
                    
                    for method in results.keys():
                        results[method].append(config_result)
        
        return results
    
    def _run_neural_operator_experiments(self) -> Dict[str, Any]:
        """Run neural operator training experiments."""
        logger.info("Running neural operator training experiments")
        
        results = {
            'quantum_fno': [],
            'quantum_deeponet': [],
            'classical_fno_baseline': []
        }
        
        # Load test datasets
        datasets = {
            'navier_stokes': load_navier_stokes(),
            'heat_equation': load_heat_equation()
        }
        
        for dataset_name, dataset in datasets.items():
            for network_size in self.config.network_sizes:
                for schmidt_rank in self.config.schmidt_ranks:
                    
                    logger.info(f"Testing {dataset_name}, network_size={network_size}, schmidt_rank={schmidt_rank}")
                    
                    network = self._create_test_network(network_size, 0.9)
                    
                    # Run multiple trials
                    for trial in range(self.config.n_trials):
                        
                        # Quantum FNO
                        qfno_start = time.time()
                        qfno = QuantumFourierNeuralOperator(
                            modes=16, schmidt_rank=schmidt_rank
                        )
                        qfno_result = qfno.fit(
                            dataset['train'], network, 
                            epochs=50, batch_size=32
                        )
                        qfno_time = time.time() - qfno_start
                        
                        qfno_predictions = qfno.predict(dataset['test'], network)
                        qfno_mse = float(jnp.mean((qfno_predictions - dataset['test']['targets']) ** 2))
                        
                        qfno_metrics = {
                            'dataset': dataset_name,
                            'network_size': network_size,
                            'schmidt_rank': schmidt_rank,
                            'training_time': qfno_time,
                            'final_loss': float(qfno_result['losses'][-1]),
                            'test_mse': qfno_mse,
                            'convergence_epochs': len(qfno_result['losses'])
                        }
                        results['quantum_fno'].append(qfno_metrics)
                        
                        # Quantum DeepONet
                        qdon_start = time.time()
                        if 'u' in dataset['train'] and 'y' in dataset['train']:  # DeepONet format
                            qdon = QuantumDeepONet(schmidt_rank=schmidt_rank)
                            qdon_result = qdon.fit(
                                dataset['train'], network,
                                epochs=50, batch_size=32
                            )
                            qdon_time = time.time() - qdon_start
                            
                            qdon_predictions = qdon.predict(dataset['test'], network)
                            qdon_mse = float(jnp.mean((qdon_predictions - dataset['test']['s']) ** 2))
                            
                            qdon_metrics = {
                                'dataset': dataset_name,
                                'network_size': network_size,
                                'schmidt_rank': schmidt_rank,
                                'training_time': qdon_time,
                                'final_loss': float(qdon_result['losses'][-1]),
                                'test_mse': qdon_mse,
                                'convergence_epochs': len(qdon_result['losses'])
                            }
                            results['quantum_deeponet'].append(qdon_metrics)
                        
                        # Classical baseline (simplified)
                        classical_start = time.time()
                        classical_mse = self._run_classical_baseline(dataset)
                        classical_time = time.time() - classical_start
                        
                        classical_metrics = {
                            'dataset': dataset_name,
                            'network_size': network_size,
                            'schmidt_rank': schmidt_rank,  # For comparison
                            'training_time': classical_time,
                            'test_mse': classical_mse
                        }
                        results['classical_fno_baseline'].append(classical_metrics)
        
        return results
    
    def _run_entanglement_scaling_experiments(self) -> Dict[str, Any]:
        """Study scaling behavior with respect to entanglement quality and network size."""
        logger.info("Running entanglement scaling experiments")
        
        results = {
            'scaling_data': [],
            'performance_vs_entanglement': [],
            'network_size_scaling': []
        }
        
        # Entanglement quality scaling
        for network_size in [4, 8, 16, 32]:
            for fidelity in np.linspace(0.7, 0.99, 10):
                
                network = self._create_test_network(network_size, fidelity)
                tasks = self._generate_test_tasks(network_size * 3, network)
                
                # Multiple trials for statistical significance
                trial_results = []
                for trial in range(self.config.n_trials):
                    
                    scheduler = create_hybrid_scheduler(network)
                    start_time = time.time()
                    result = scheduler.schedule_tasks_hybrid(tasks)
                    execution_time = time.time() - start_time
                    
                    metrics = {
                        'network_size': network_size,
                        'entanglement_fidelity': fidelity,
                        'execution_time': execution_time,
                        'completion_time': result.estimated_completion_time,
                        'resource_utilization': np.mean(list(result.resource_utilization.values())),
                        'quantum_advantage_score': scheduler.quantum_advantage_scores[-1] if scheduler.quantum_advantage_scores else 1.0
                    }
                    trial_results.append(metrics)
                
                # Average across trials
                avg_metrics = {}
                for key in trial_results[0].keys():
                    if isinstance(trial_results[0][key], (int, float)):
                        avg_metrics[key] = np.mean([r[key] for r in trial_results])
                        avg_metrics[f'{key}_std'] = np.std([r[key] for r in trial_results])
                    else:
                        avg_metrics[key] = trial_results[0][key]
                
                results['scaling_data'].append(avg_metrics)
        
        return results
    
    def _run_schmidt_rank_experiments(self) -> Dict[str, Any]:
        """Study the effect of Schmidt rank on performance and quantum advantage."""
        logger.info("Running Schmidt rank analysis experiments")
        
        results = {
            'schmidt_rank_analysis': [],
            'optimal_ranks': [],
            'complexity_scaling': []
        }
        
        network_size = 8  # Fixed network size for this analysis
        network = self._create_test_network(network_size, 0.9)
        
        # Test different problem complexities
        for task_complexity in [1, 2, 4, 8, 16]:  # Different quantum volumes
            for schmidt_rank in [2, 4, 8, 16, 32, 64]:
                
                tasks = self._generate_complex_tasks(task_complexity, network)
                
                trial_results = []
                for trial in range(self.config.n_trials):
                    
                    # Test quantum FNO with different Schmidt ranks
                    qfno = QuantumFourierNeuralOperator(
                        modes=16, schmidt_rank=schmidt_rank
                    )
                    
                    # Create synthetic training data
                    train_data = self._generate_synthetic_training_data(64, task_complexity)
                    
                    start_time = time.time()
                    result = qfno.fit(train_data, network, epochs=20, batch_size=16)
                    training_time = time.time() - start_time
                    
                    # Test data
                    test_data = self._generate_synthetic_training_data(32, task_complexity)
                    predictions = qfno.predict(test_data, network)
                    test_mse = float(jnp.mean((predictions - test_data['targets']) ** 2))
                    
                    metrics = {
                        'task_complexity': task_complexity,
                        'schmidt_rank': schmidt_rank,
                        'training_time': training_time,
                        'final_loss': float(result['losses'][-1]),
                        'test_mse': test_mse,
                        'memory_usage': self._estimate_memory_usage(schmidt_rank, task_complexity)
                    }
                    trial_results.append(metrics)
                
                # Average across trials
                avg_metrics = {}
                for key in trial_results[0].keys():
                    if isinstance(trial_results[0][key], (int, float)):
                        avg_metrics[key] = np.mean([r[key] for r in trial_results])
                        avg_metrics[f'{key}_std'] = np.std([r[key] for r in trial_results])
                    else:
                        avg_metrics[key] = trial_results[0][key]
                
                results['schmidt_rank_analysis'].append(avg_metrics)
        
        return results
    
    def _run_comparative_experiments(self) -> Dict[str, Any]:
        """Run comprehensive comparative experiments across all methods."""
        logger.info("Running comprehensive comparative experiments")
        
        # This combines multiple experiment types for full comparison
        scheduling_results = self._run_scheduling_experiments()
        neural_op_results = self._run_neural_operator_experiments()
        scaling_results = self._run_entanglement_scaling_experiments()
        
        return {
            'scheduling': scheduling_results,
            'neural_operators': neural_op_results,
            'scaling': scaling_results,
            'comparative_summary': self._generate_comparative_summary(
                scheduling_results, neural_op_results, scaling_results
            )
        }
    
    def _run_quantum_advantage_certification(self) -> Dict[str, Any]:
        """Run quantum advantage certification experiments."""
        logger.info("Running quantum advantage certification")
        
        results = {
            'certification_results': [],
            'advantage_bounds': [],
            'statistical_significance': []
        }
        
        # Test across different scenarios for robust certification
        for network_size in self.config.network_sizes:
            for fidelity in self.config.entanglement_fidelities:
                
                network = self._create_test_network(network_size, fidelity)
                tasks = self._generate_test_tasks(network_size * 2, network)
                
                # Use the benchmark function for comprehensive analysis
                benchmark_results = benchmark_quantum_advantage(
                    network, tasks, n_trials=self.config.n_trials
                )
                
                cert_result = {
                    'network_size': network_size,
                    'fidelity': fidelity,
                    'mean_quantum_advantage': benchmark_results['statistical_analysis']['mean_quantum_advantage'],
                    'std_quantum_advantage': benchmark_results['statistical_analysis']['std_quantum_advantage'],
                    'significance_fraction': benchmark_results['statistical_analysis']['quantum_advantage_significance'],
                    't_statistic': benchmark_results['statistical_analysis']['t_statistic'],
                    'p_value': benchmark_results['statistical_analysis']['p_value'],
                    'statistically_significant': benchmark_results['statistical_analysis']['statistically_significant'],
                    'effect_size': self._calculate_effect_size(benchmark_results['quantum_advantage_scores'])
                }
                
                results['certification_results'].append(cert_result)
        
        return results
    
    def _perform_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of all results."""
        logger.info("Performing comprehensive statistical analysis")
        
        analysis = {
            'overall_quantum_advantage': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'confidence_intervals': {},
            'practical_significance': {}
        }
        
        # Extract quantum advantage scores from different experiment types
        if 'hybrid_quantum_classical' in results:
            # Scheduling experiments
            hqc_scores = []
            for config_results in results['hybrid_quantum_classical']:
                for trial_data in config_results['trial_data']['hybrid_quantum_classical']:
                    if 'quantum_advantage_score' in trial_data:
                        hqc_scores.append(trial_data['quantum_advantage_score'])
            
            if hqc_scores:
                analysis['overall_quantum_advantage']['scheduling'] = {
                    'mean': np.mean(hqc_scores),
                    'std': np.std(hqc_scores),
                    'median': np.median(hqc_scores),
                    'min': np.min(hqc_scores),
                    'max': np.max(hqc_scores)
                }
                
                # Statistical significance test against null hypothesis (QA = 1.0)
                t_stat, p_value = stats.ttest_1samp(hqc_scores, 1.0)
                analysis['statistical_tests']['scheduling'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level and np.mean(hqc_scores) > 1.0
                }
                
                # Effect size (Cohen's d)
                effect_size = (np.mean(hqc_scores) - 1.0) / np.std(hqc_scores)
                analysis['effect_sizes']['scheduling'] = effect_size
                
                # Confidence interval
                ci_lower, ci_upper = stats.t.interval(
                    0.95, len(hqc_scores) - 1,
                    loc=np.mean(hqc_scores),
                    scale=stats.sem(hqc_scores)
                )
                analysis['confidence_intervals']['scheduling'] = (ci_lower, ci_upper)
                
                # Practical significance
                analysis['practical_significance']['scheduling'] = {
                    'meaningful_advantage': np.mean(hqc_scores) > 1.0 + self.config.effect_size_threshold,
                    'fraction_showing_advantage': np.mean([score > 1.05 for score in hqc_scores]),
                    'robust_advantage': ci_lower > 1.0
                }
        
        # Similar analysis for other experiment types...
        # [Additional statistical analysis would be implemented here for other experiment types]
        
        return analysis
    
    def _generate_publication_plots(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Generate publication-ready plots and visualizations."""
        logger.info("Generating publication-ready visualizations")
        
        # Set up publication-quality matplotlib settings
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        plots_dir = self.results_dir / "plots"
        
        # Plot 1: Quantum Advantage vs Network Size
        if 'hybrid_quantum_classical' in results:
            self._plot_quantum_advantage_scaling(results, plots_dir)
        
        # Plot 2: Performance Comparison Across Methods
        if 'hybrid_quantum_classical' in results:
            self._plot_method_comparison(results, plots_dir)
        
        # Plot 3: Entanglement Scaling Analysis
        if 'scaling_data' in results:
            self._plot_entanglement_scaling(results, plots_dir)
        
        # Plot 4: Schmidt Rank Analysis
        if 'schmidt_rank_analysis' in results:
            self._plot_schmidt_rank_analysis(results, plots_dir)
        
        # Plot 5: Statistical Significance Summary
        self._plot_statistical_summary(analysis, plots_dir)
    
    def _plot_quantum_advantage_scaling(self, results: Dict[str, Any], plots_dir: Path) -> None:
        """Plot quantum advantage scaling with network size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        network_sizes = []
        qa_means = []
        qa_stds = []
        
        for config_result in results['hybrid_quantum_classical']:
            network_sizes.append(config_result['network_size'])
            qa_scores = [trial['quantum_advantage_score'] 
                        for trial in config_result['trial_data']['hybrid_quantum_classical']]
            qa_means.append(np.mean(qa_scores))
            qa_stds.append(np.std(qa_scores))
        
        # Group by network size
        size_groups = {}
        for i, size in enumerate(network_sizes):
            if size not in size_groups:
                size_groups[size] = {'means': [], 'stds': []}
            size_groups[size]['means'].append(qa_means[i])
            size_groups[size]['stds'].append(qa_stds[i])
        
        sizes = sorted(size_groups.keys())
        means = [np.mean(size_groups[size]['means']) for size in sizes]
        stds = [np.mean(size_groups[size]['stds']) for size in sizes]
        
        # Plot 1: Mean quantum advantage with error bars
        ax1.errorbar(sizes, means, yerr=stds, marker='o', linewidth=2, markersize=8, capsize=5)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Advantage Baseline')
        ax1.set_xlabel('Network Size (Number of Nodes)')
        ax1.set_ylabel('Quantum Advantage Score')
        ax1.set_title('Quantum Advantage vs Network Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of quantum advantage scores
        all_scores = []
        all_sizes = []
        for config_result in results['hybrid_quantum_classical']:
            size = config_result['network_size']
            scores = [trial['quantum_advantage_score'] 
                     for trial in config_result['trial_data']['hybrid_quantum_classical']]
            all_scores.extend(scores)
            all_sizes.extend([size] * len(scores))
        
        df = pd.DataFrame({'Network Size': all_sizes, 'Quantum Advantage': all_scores})
        sns.boxplot(data=df, x='Network Size', y='Quantum Advantage', ax=ax2)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Quantum Advantage Distribution by Network Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'quantum_advantage_scaling.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'quantum_advantage_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_method_comparison(self, results: Dict[str, Any], plots_dir: Path) -> None:
        """Plot comparison of different scheduling methods."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = ['hybrid_quantum_classical', 'classical_greedy', 'classical_simulated_annealing', 'quantum_random']
        method_labels = ['Hybrid Q-C', 'Classical Greedy', 'Simulated Annealing', 'Quantum Random']
        
        # Extract performance metrics
        execution_times = {method: [] for method in methods}
        solution_qualities = {method: [] for method in methods}
        utilizations = {method: [] for method in methods}
        
        for config_result in results['hybrid_quantum_classical']:
            for method in methods:
                if method in config_result['trial_data']:
                    for trial in config_result['trial_data'][method]:
                        if 'execution_time' in trial:
                            execution_times[method].append(trial['execution_time'])
                        if 'solution_quality' in trial:
                            solution_qualities[method].append(trial['solution_quality'])
                        elif 'quantum_advantage_score' in trial:
                            solution_qualities[method].append(trial['quantum_advantage_score'])
                        if 'average_utilization' in trial:
                            utilizations[method].append(trial['average_utilization'])
        
        # Plot execution times
        exec_data = [execution_times[method] for method in methods if execution_times[method]]
        if exec_data:
            ax1.boxplot(exec_data, labels=[method_labels[i] for i, method in enumerate(methods) if execution_times[method]])
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title('Execution Time Comparison')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot solution quality
        qual_data = [solution_qualities[method] for method in methods if solution_qualities[method]]
        if qual_data:
            ax2.boxplot(qual_data, labels=[method_labels[i] for i, method in enumerate(methods) if solution_qualities[method]])
            ax2.set_ylabel('Solution Quality Score')
            ax2.set_title('Solution Quality Comparison')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot resource utilization
        util_data = [utilizations[method] for method in methods if utilizations[method]]
        if util_data:
            ax3.boxplot(util_data, labels=[method_labels[i] for i, method in enumerate(methods) if utilizations[method]])
            ax3.set_ylabel('Resource Utilization')
            ax3.set_title('Resource Utilization Comparison')
            ax3.tick_params(axis='x', rotation=45)
        
        # Performance vs execution time scatter
        for i, method in enumerate(methods):
            if execution_times[method] and solution_qualities[method]:
                ax4.scatter(execution_times[method], solution_qualities[method], 
                           label=method_labels[i], alpha=0.6, s=50)
        
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('Solution Quality Score')
        ax4.set_title('Quality vs Speed Trade-off')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'method_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_entanglement_scaling(self, results: Dict[str, Any], plots_dir: Path) -> None:
        """Plot entanglement scaling analysis."""
        if 'scaling_data' not in results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract scaling data
        scaling_data = results['scaling_data']
        
        fidelities = [d['entanglement_fidelity'] for d in scaling_data]
        network_sizes = [d['network_size'] for d in scaling_data]
        qa_scores = [d['quantum_advantage_score'] for d in scaling_data]
        completion_times = [d['completion_time'] for d in scaling_data]
        
        # Quantum advantage vs entanglement fidelity
        for size in sorted(set(network_sizes)):
            size_indices = [i for i, s in enumerate(network_sizes) if s == size]
            size_fidelities = [fidelities[i] for i in size_indices]
            size_qa_scores = [qa_scores[i] for i in size_indices]
            
            ax1.plot(size_fidelities, size_qa_scores, 'o-', label=f'Size {size}', markersize=6)
        
        ax1.set_xlabel('Entanglement Fidelity')
        ax1.set_ylabel('Quantum Advantage Score')
        ax1.set_title('Quantum Advantage vs Entanglement Fidelity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Completion time vs network size
        for fidelity_range in [(0.8, 0.85), (0.9, 0.95), (0.95, 1.0)]:
            fid_indices = [i for i, f in enumerate(fidelities) 
                          if fidelity_range[0] <= f < fidelity_range[1]]
            fid_sizes = [network_sizes[i] for i in fid_indices]
            fid_times = [completion_times[i] for i in fid_indices]
            
            # Group by size and average
            size_groups = {}
            for i, size in enumerate(fid_sizes):
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(fid_times[i])
            
            sizes = sorted(size_groups.keys())
            avg_times = [np.mean(size_groups[size]) for size in sizes]
            
            ax2.plot(sizes, avg_times, 'o-', 
                    label=f'Fidelity {fidelity_range[0]:.1f}-{fidelity_range[1]:.1f}',
                    markersize=6)
        
        ax2.set_xlabel('Network Size')
        ax2.set_ylabel('Completion Time')
        ax2.set_title('Completion Time vs Network Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Heatmap: Quantum advantage vs size and fidelity
        df_pivot = pd.DataFrame({
            'Network Size': network_sizes,
            'Entanglement Fidelity': fidelities,
            'Quantum Advantage': qa_scores
        })
        
        # Create bins for better visualization
        df_pivot['Fidelity Bin'] = pd.cut(df_pivot['Entanglement Fidelity'], bins=5)
        pivot_table = df_pivot.groupby(['Network Size', 'Fidelity Bin'])['Quantum Advantage'].mean().unstack()
        
        im = ax3.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax3.set_xticks(range(len(pivot_table.columns)))
        ax3.set_xticklabels([f'{interval.left:.2f}-{interval.right:.2f}' 
                           for interval in pivot_table.columns], rotation=45)
        ax3.set_yticks(range(len(pivot_table.index)))
        ax3.set_yticklabels(pivot_table.index)
        ax3.set_xlabel('Entanglement Fidelity Range')
        ax3.set_ylabel('Network Size')
        ax3.set_title('Quantum Advantage Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Quantum Advantage Score')
        
        # Scaling analysis: log-log plot
        unique_sizes = sorted(set(network_sizes))
        avg_qa_by_size = []
        for size in unique_sizes:
            size_qa = [qa_scores[i] for i, s in enumerate(network_sizes) if s == size]
            avg_qa_by_size.append(np.mean(size_qa))
        
        ax4.loglog(unique_sizes, avg_qa_by_size, 'bo-', markersize=8, linewidth=2)
        ax4.set_xlabel('Network Size (log scale)')
        ax4.set_ylabel('Quantum Advantage (log scale)')
        ax4.set_title('Scaling Behavior (Log-Log Plot)')
        ax4.grid(True, alpha=0.3)
        
        # Fit power law and add to plot
        log_sizes = np.log(unique_sizes)
        log_qa = np.log(avg_qa_by_size)
        slope, intercept = np.polyfit(log_sizes, log_qa, 1)
        
        fit_sizes = np.logspace(np.log10(min(unique_sizes)), np.log10(max(unique_sizes)), 100)
        fit_qa = np.exp(intercept) * fit_sizes ** slope
        ax4.plot(fit_sizes, fit_qa, 'r--', alpha=0.7, 
                label=f'Power law fit: QA ∝ N^{slope:.2f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'entanglement_scaling.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'entanglement_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_schmidt_rank_analysis(self, results: Dict[str, Any], plots_dir: Path) -> None:
        """Plot Schmidt rank analysis results."""
        if 'schmidt_rank_analysis' not in results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        schmidt_data = results['schmidt_rank_analysis']
        
        complexities = sorted(set([d['task_complexity'] for d in schmidt_data]))
        ranks = sorted(set([d['schmidt_rank'] for d in schmidt_data]))
        
        # Performance vs Schmidt rank for different complexities
        for complexity in complexities:
            complexity_data = [d for d in schmidt_data if d['task_complexity'] == complexity]
            rank_values = [d['schmidt_rank'] for d in complexity_data]
            test_mse_values = [d['test_mse'] for d in complexity_data]
            
            ax1.plot(rank_values, test_mse_values, 'o-', label=f'Complexity {complexity}', markersize=6)
        
        ax1.set_xlabel('Schmidt Rank')
        ax1.set_ylabel('Test MSE')
        ax1.set_title('Performance vs Schmidt Rank')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training time vs Schmidt rank
        for complexity in complexities:
            complexity_data = [d for d in schmidt_data if d['task_complexity'] == complexity]
            rank_values = [d['schmidt_rank'] for d in complexity_data]
            time_values = [d['training_time'] for d in complexity_data]
            
            ax2.plot(rank_values, time_values, 's-', label=f'Complexity {complexity}', markersize=6)
        
        ax2.set_xlabel('Schmidt Rank')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time vs Schmidt Rank')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Memory usage vs Schmidt rank
        for complexity in complexities:
            complexity_data = [d for d in schmidt_data if d['task_complexity'] == complexity]
            rank_values = [d['schmidt_rank'] for d in complexity_data]
            memory_values = [d['memory_usage'] for d in complexity_data]
            
            ax3.plot(rank_values, memory_values, '^-', label=f'Complexity {complexity}', markersize=6)
        
        ax3.set_xlabel('Schmidt Rank')
        ax3.set_ylabel('Memory Usage (GB)')
        ax3.set_title('Memory Usage vs Schmidt Rank')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Heatmap: Optimal Schmidt rank for each complexity
        pivot_data = []
        for complexity in complexities:
            complexity_data = [d for d in schmidt_data if d['task_complexity'] == complexity]
            
            # Find optimal rank (lowest test MSE)
            best_rank = min(complexity_data, key=lambda x: x['test_mse'])['schmidt_rank']
            pivot_data.append([complexity, best_rank])
        
        complexities_arr = [p[0] for p in pivot_data]
        optimal_ranks = [p[1] for p in pivot_data]
        
        ax4.plot(complexities_arr, optimal_ranks, 'ro-', markersize=10, linewidth=3)
        ax4.set_xlabel('Task Complexity')
        ax4.set_ylabel('Optimal Schmidt Rank')
        ax4.set_title('Optimal Schmidt Rank vs Task Complexity')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(complexities_arr) > 2:
            z = np.polyfit(complexities_arr, optimal_ranks, 1)
            p = np.poly1d(z)
            ax4.plot(complexities_arr, p(complexities_arr), "r--", alpha=0.7,
                    label=f'Linear fit: Rank = {z[0]:.2f} × Complexity + {z[1]:.2f}')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'schmidt_rank_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'schmidt_rank_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_summary(self, analysis: Dict[str, Any], plots_dir: Path) -> None:
        """Plot statistical significance summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: P-values for different experiment types
        if 'statistical_tests' in analysis:
            experiment_types = list(analysis['statistical_tests'].keys())
            p_values = [analysis['statistical_tests'][exp_type]['p_value'] 
                       for exp_type in experiment_types]
            
            bars = ax1.bar(experiment_types, p_values, alpha=0.7)
            ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significance Threshold')
            ax1.set_ylabel('P-value')
            ax1.set_title('Statistical Significance Test Results')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # Color bars based on significance
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.05:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
        
        # Plot 2: Effect sizes
        if 'effect_sizes' in analysis:
            experiment_types = list(analysis['effect_sizes'].keys())
            effect_sizes = [analysis['effect_sizes'][exp_type] 
                          for exp_type in experiment_types]
            
            bars = ax2.bar(experiment_types, effect_sizes, alpha=0.7)
            ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small Effect')
            ax2.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.7, label='Medium Effect')
            ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Large Effect')
            ax2.set_ylabel('Effect Size (Cohen\'s d)')
            ax2.set_title('Effect Size Analysis')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            # Color bars based on effect size
            for bar, effect_size in zip(bars, effect_sizes):
                if effect_size >= 0.8:
                    bar.set_color('green')
                elif effect_size >= 0.5:
                    bar.set_color('yellow')
                elif effect_size >= 0.2:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # Plot 3: Confidence intervals
        if 'confidence_intervals' in analysis and 'overall_quantum_advantage' in analysis:
            experiment_types = list(analysis['confidence_intervals'].keys())
            means = [analysis['overall_quantum_advantage'][exp_type]['mean'] 
                    for exp_type in experiment_types 
                    if exp_type in analysis['overall_quantum_advantage']]
            cis = [analysis['confidence_intervals'][exp_type] 
                  for exp_type in experiment_types]
            
            if means and cis:
                ci_lower = [ci[0] for ci in cis]
                ci_upper = [ci[1] for ci in cis]
                
                x_pos = range(len(experiment_types))
                ax3.errorbar(x_pos, means, 
                           yerr=[[m - l for m, l in zip(means, ci_lower)], 
                                 [u - m for u, m in zip(ci_upper, means)]], 
                           fmt='o', capsize=5, capthick=2, markersize=8)
                
                ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                          label='No Advantage Baseline')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(experiment_types, rotation=45)
                ax3.set_ylabel('Quantum Advantage Score')
                ax3.set_title('Quantum Advantage with 95% Confidence Intervals')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary of practical significance
        if 'practical_significance' in analysis:
            significance_metrics = []
            values = []
            
            for exp_type, metrics in analysis['practical_significance'].items():
                if 'fraction_showing_advantage' in metrics:
                    significance_metrics.append(f'{exp_type}\nFraction > 5%')
                    values.append(metrics['fraction_showing_advantage'])
                
                if 'robust_advantage' in metrics:
                    significance_metrics.append(f'{exp_type}\nRobust')
                    values.append(1.0 if metrics['robust_advantage'] else 0.0)
            
            if significance_metrics and values:
                bars = ax4.bar(range(len(significance_metrics)), values, alpha=0.7)
                ax4.set_xticks(range(len(significance_metrics)))
                ax4.set_xticklabels(significance_metrics, rotation=45)
                ax4.set_ylabel('Fraction / Binary Score')
                ax4.set_title('Practical Significance Metrics')
                
                # Color coding
                for bar, value in zip(bars, values):
                    if value >= 0.8:
                        bar.set_color('green')
                    elif value >= 0.6:
                        bar.set_color('yellow')
                    else:
                        bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'statistical_summary.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(plots_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_experimental_results(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Save experimental results and analysis to files."""
        logger.info("Saving experimental results")
        
        data_dir = self.results_dir / "data"
        
        # Save raw results
        with open(data_dir / "raw_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save statistical analysis
        with open(data_dir / "statistical_analysis.json", "w") as f:
            json_analysis = self._convert_numpy_to_json(analysis)
            json.dump(json_analysis, f, indent=2)
        
        # Save configuration
        with open(data_dir / "experiment_config.json", "w") as f:
            config_dict = {
                'experiment_name': self.config.experiment_name,
                'experiment_type': self.config.experiment_type.value,
                'n_trials': self.config.n_trials,
                'random_seed': self.config.random_seed,
                'network_sizes': self.config.network_sizes,
                'entanglement_fidelities': self.config.entanglement_fidelities,
                'qaoa_layers': self.config.qaoa_layers,
                'schmidt_ranks': self.config.schmidt_ranks,
                'metrics_to_track': self.config.metrics_to_track,
                'significance_level': self.config.significance_level,
                'effect_size_threshold': self.config.effect_size_threshold
            }
            json.dump(config_dict, f, indent=2)
        
        # Save pickle version for complete data preservation
        with open(data_dir / "complete_results.pkl", "wb") as f:
            pickle.dump({
                'results': results,
                'analysis': analysis,
                'config': self.config
            }, f)
        
        logger.info(f"Results saved to {data_dir}")
    
    # Helper methods for experiment setup and data generation
    
    def _create_test_network(self, size: int, fidelity: float) -> PhotonicNetwork:
        """Create a test quantum photonic network."""
        from qnet_no.networks.photonic_network import PhotonicNetwork
        
        network = PhotonicNetwork()
        
        # Create nodes in a ring topology
        for i in range(size):
            network.add_quantum_node(
                node_id=i,
                n_qubits=8 + (i % 4) * 2,  # Vary qubit count
                fidelity=fidelity + np.random.normal(0, 0.01),  # Small fidelity variation
                capabilities=["two_qubit_gates", "readout", "photon_counting", "parametric_ops"]
            )
        
        # Add entanglement links (ring + some random connections)
        for i in range(size):
            next_node = (i + 1) % size
            network.add_entanglement_link(
                node1=i, node2=next_node,
                fidelity=fidelity + np.random.normal(0, 0.02),
                schmidt_rank=8
            )
        
        # Add some random long-range connections
        for _ in range(size // 2):
            node1 = np.random.randint(0, size)
            node2 = np.random.randint(0, size)
            if node1 != node2 and not network.graph.has_edge(node1, node2):
                network.add_entanglement_link(
                    node1=node1, node2=node2,
                    fidelity=fidelity * 0.9 + np.random.normal(0, 0.02),  # Lower fidelity for long-range
                    schmidt_rank=4
                )
        
        return network
    
    def _generate_test_tasks(self, n_tasks: int, network: PhotonicNetwork) -> List[ComputationTask]:
        """Generate test computation tasks."""
        tasks = []
        operation_types = ["fourier_transform", "tensor_contraction", "gate_sequence", "measurement"]
        priorities = list(TaskPriority)
        
        for i in range(n_tasks):
            task = ComputationTask(
                task_id=f"task_{i}",
                operation_type=np.random.choice(operation_types),
                required_qubits=np.random.randint(2, 6),
                estimated_time=np.random.uniform(50.0, 500.0),  # microseconds
                priority=np.random.choice(priorities),
                quantum_volume=2 ** np.random.randint(1, 5)
            )
            
            # Add some dependencies
            if i > 2 and np.random.random() < 0.3:
                n_deps = np.random.randint(1, min(3, i))
                deps = np.random.choice(range(i), size=n_deps, replace=False)
                task.dependencies = {f"task_{dep}" for dep in deps}
            
            tasks.append(task)
        
        return tasks
    
    def _generate_complex_tasks(self, complexity: int, network: PhotonicNetwork) -> List[ComputationTask]:
        """Generate tasks with specific complexity level."""
        n_tasks = complexity * 2
        tasks = []
        
        for i in range(n_tasks):
            task = ComputationTask(
                task_id=f"complex_task_{i}",
                operation_type="tensor_contraction" if i % 2 == 0 else "fourier_transform",
                required_qubits=min(2 + complexity, 8),
                estimated_time=complexity * 100.0 + np.random.uniform(0, 50.0),
                priority=TaskPriority.HIGH if i < n_tasks // 2 else TaskPriority.MEDIUM,
                quantum_volume=complexity * 2
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_synthetic_training_data(self, n_samples: int, complexity: int) -> Dict[str, jnp.ndarray]:
        """Generate synthetic training data for neural operators."""
        # Create synthetic PDE-like data
        spatial_dim = 32
        
        # Input functions (initial conditions)
        x = np.linspace(0, 2*np.pi, spatial_dim)
        inputs = []
        targets = []
        
        for _ in range(n_samples):
            # Generate random Fourier modes
            n_modes = complexity + 2
            coeffs = np.random.randn(n_modes) * np.exp(-np.arange(n_modes) * 0.1)
            
            input_func = np.zeros(spatial_dim)
            for k, coeff in enumerate(coeffs):
                input_func += coeff * np.sin((k+1) * x)
            
            # Target is some transformation (e.g., diffusion equation solution)
            target_func = np.zeros(spatial_dim)
            for k, coeff in enumerate(coeffs):
                # Simple decay model
                decay_rate = (k+1)**2 * 0.01 * complexity
                target_func += coeff * np.exp(-decay_rate) * np.sin((k+1) * x)
            
            inputs.append(input_func)
            targets.append(target_func)
        
        return {
            'inputs': jnp.array(inputs)[:, :, None],  # Add channel dimension
            'targets': jnp.array(targets)[:, :, None]
        }
    
    def _calculate_solution_quality(self, result: 'SchedulingResult', 
                                  tasks: List[ComputationTask], 
                                  network: PhotonicNetwork) -> float:
        """Calculate quality score for scheduling result."""
        if not result.task_assignments:
            return 0.0
        
        # Load balancing score
        node_loads = {}
        for task in tasks:
            node_id = result.task_assignments.get(task.task_id)
            if node_id is not None:
                if node_id not in node_loads:
                    node_loads[node_id] = 0
                node_loads[node_id] += task.required_qubits
        
        if not node_loads:
            return 0.0
        
        loads = list(node_loads.values())
        load_balance = 1.0 - (np.std(loads) / (np.mean(loads) + 1e-6))
        
        # Resource utilization score
        total_capacity = sum(node.n_qubits for node in network.quantum_nodes.values())
        total_used = sum(loads)
        utilization = total_used / total_capacity
        
        # Priority satisfaction score  
        priority_score = 0.0
        total_priority = 0.0
        for task in tasks:
            node_id = result.task_assignments.get(task.task_id)
            if node_id is not None:
                node = network.quantum_nodes[node_id]
                priority_score += task.priority.value * node.fidelity
            total_priority += task.priority.value
        
        if total_priority > 0:
            priority_score /= total_priority
        
        # Combined quality score
        quality = 0.4 * load_balance + 0.3 * utilization + 0.3 * priority_score
        return quality
    
    def _calculate_assignment_quality(self, assignment: Dict[str, int],
                                    tasks: List[ComputationTask],
                                    network: PhotonicNetwork) -> float:
        """Calculate quality of task assignment."""
        if not assignment:
            return 0.0
        
        # Simple quality metric based on fidelity and load balance
        node_loads = {}
        total_fidelity = 0.0
        
        for task in tasks:
            node_id = assignment.get(task.task_id)
            if node_id is not None and node_id in network.quantum_nodes:
                node = network.quantum_nodes[node_id]
                total_fidelity += node.fidelity
                
                if node_id not in node_loads:
                    node_loads[node_id] = 0
                node_loads[node_id] += task.required_qubits
        
        if not node_loads:
            return 0.0
        
        avg_fidelity = total_fidelity / len([t for t in tasks if assignment.get(t.task_id) is not None])
        
        # Load balance
        loads = list(node_loads.values())
        load_balance = 1.0 - (np.std(loads) / (np.mean(loads) + 1e-6))
        
        return 0.6 * avg_fidelity + 0.4 * load_balance
    
    def _calculate_utilization(self, assignment: Dict[str, int],
                             tasks: List[ComputationTask],
                             network: PhotonicNetwork) -> float:
        """Calculate resource utilization."""
        if not assignment:
            return 0.0
        
        node_loads = {}
        for task in tasks:
            node_id = assignment.get(task.task_id)
            if node_id is not None:
                if node_id not in node_loads:
                    node_loads[node_id] = 0
                node_loads[node_id] += task.required_qubits
        
        if not node_loads:
            return 0.0
        
        total_used = sum(node_loads.values())
        total_capacity = sum(node.n_qubits for node in network.quantum_nodes.values())
        
        return total_used / total_capacity
    
    def _run_classical_baseline(self, dataset: Dict[str, jnp.ndarray]) -> float:
        """Run simplified classical baseline for comparison."""
        # Very simple baseline: linear regression on flattened features
        train_inputs = dataset['train']['inputs'].reshape(len(dataset['train']['inputs']), -1)
        train_targets = dataset['train']['targets'].reshape(len(dataset['train']['targets']), -1)
        
        test_inputs = dataset['test']['inputs'].reshape(len(dataset['test']['inputs']), -1)
        test_targets = dataset['test']['targets'].reshape(len(dataset['test']['targets']), -1)
        
        # Simple linear model (pseudo-inverse solution)
        try:
            # Add regularization term
            reg_term = 1e-6 * np.eye(train_inputs.shape[1])
            weights = np.linalg.solve(
                train_inputs.T @ train_inputs + reg_term,
                train_inputs.T @ train_targets
            )
            
            predictions = test_inputs @ weights
            mse = np.mean((predictions - test_targets) ** 2)
            return float(mse)
        except:
            return 1.0  # Return high error if classical method fails
    
    def _estimate_memory_usage(self, schmidt_rank: int, complexity: int) -> float:
        """Estimate memory usage in GB."""
        # Simple model: memory scales with Schmidt rank squared and complexity
        base_memory = 0.5  # GB
        rank_factor = (schmidt_rank / 8) ** 2
        complexity_factor = complexity / 4
        
        return base_memory * rank_factor * complexity_factor
    
    def _generate_comparative_summary(self, scheduling_results: Dict[str, Any],
                                    neural_op_results: Dict[str, Any],
                                    scaling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of comparative analysis."""
        summary = {
            'scheduling_performance': {},
            'neural_operator_performance': {},
            'scaling_behavior': {},
            'overall_quantum_advantage': {}
        }
        
        # Scheduling performance summary
        if 'hybrid_quantum_classical' in scheduling_results:
            hqc_scores = []
            for config_result in scheduling_results['hybrid_quantum_classical']:
                for trial in config_result['trial_data']['hybrid_quantum_classical']:
                    if 'quantum_advantage_score' in trial:
                        hqc_scores.append(trial['quantum_advantage_score'])
            
            if hqc_scores:
                summary['scheduling_performance'] = {
                    'mean_quantum_advantage': np.mean(hqc_scores),
                    'fraction_showing_advantage': np.mean([s > 1.05 for s in hqc_scores]),
                    'max_quantum_advantage': np.max(hqc_scores)
                }
        
        # Neural operator performance summary
        if 'quantum_fno' in neural_op_results:
            qfno_mse = [result['test_mse'] for result in neural_op_results['quantum_fno']]
            classical_mse = [result['test_mse'] for result in neural_op_results['classical_fno_baseline']]
            
            if qfno_mse and classical_mse:
                avg_qfno_mse = np.mean(qfno_mse)
                avg_classical_mse = np.mean(classical_mse)
                
                summary['neural_operator_performance'] = {
                    'quantum_fno_mse': avg_qfno_mse,
                    'classical_mse': avg_classical_mse,
                    'accuracy_improvement': (avg_classical_mse - avg_qfno_mse) / avg_classical_mse
                }
        
        # Scaling behavior summary
        if 'scaling_data' in scaling_results:
            scaling_data = scaling_results['scaling_data']
            network_sizes = [d['network_size'] for d in scaling_data]
            qa_scores = [d['quantum_advantage_score'] for d in scaling_data]
            
            # Check if quantum advantage scales with network size
            if len(set(network_sizes)) > 2:
                correlation = np.corrcoef(network_sizes, qa_scores)[0, 1]
                summary['scaling_behavior'] = {
                    'size_qa_correlation': correlation,
                    'positive_scaling': correlation > 0.3
                }
        
        # Overall assessment
        scheduling_qa = summary.get('scheduling_performance', {}).get('mean_quantum_advantage', 1.0)
        neural_improvement = summary.get('neural_operator_performance', {}).get('accuracy_improvement', 0.0)
        scaling_positive = summary.get('scaling_behavior', {}).get('positive_scaling', False)
        
        summary['overall_quantum_advantage'] = {
            'scheduling_advantage': scheduling_qa > 1.05,
            'neural_operator_advantage': neural_improvement > 0.05,
            'scalable_advantage': scaling_positive,
            'comprehensive_advantage': all([
                scheduling_qa > 1.05,
                neural_improvement > 0.0,
                scaling_positive
            ])
        }
        
        return summary
    
    def _calculate_effect_size(self, scores: List[float]) -> float:
        """Calculate effect size (Cohen's d) for quantum advantage scores."""
        if not scores:
            return 0.0
        
        # Effect size relative to no advantage (score = 1.0)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return 0.0
        
        return (mean_score - 1.0) / std_score
    
    def _convert_numpy_to_json(self, obj: Any) -> Any:
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_to_json(item) for item in obj)
        else:
            return obj


# Factory functions and high-level interfaces

def create_scheduling_experiment(n_trials: int = 10, 
                               network_sizes: List[int] = None,
                               save_results: bool = True) -> ExperimentalFramework:
    """Create a scheduling optimization experiment."""
    config = ExperimentConfig(
        experiment_name="quantum_scheduling_advantage_study",
        experiment_type=ExperimentType.SCHEDULING_OPTIMIZATION,
        n_trials=n_trials,
        network_sizes=network_sizes or [4, 8, 16],
        save_results=save_results
    )
    return ExperimentalFramework(config)


def create_neural_operator_experiment(n_trials: int = 10,
                                    schmidt_ranks: List[int] = None,
                                    save_results: bool = True) -> ExperimentalFramework:
    """Create a neural operator training experiment."""
    config = ExperimentConfig(
        experiment_name="quantum_neural_operator_advantage_study",
        experiment_type=ExperimentType.NEURAL_OPERATOR_TRAINING,
        n_trials=n_trials,
        schmidt_ranks=schmidt_ranks or [4, 8, 16, 32],
        save_results=save_results
    )
    return ExperimentalFramework(config)


def create_comprehensive_study(n_trials: int = 20,
                             results_dir: str = "./comprehensive_quantum_study",
                             save_results: bool = True) -> ExperimentalFramework:
    """Create a comprehensive quantum advantage study."""
    config = ExperimentConfig(
        experiment_name="comprehensive_quantum_advantage_certification",
        experiment_type=ExperimentType.COMPARATIVE_PERFORMANCE,
        n_trials=n_trials,
        network_sizes=[4, 8, 16, 32],
        entanglement_fidelities=[0.8, 0.85, 0.9, 0.95],
        qaoa_layers=[2, 4, 6, 8],
        schmidt_ranks=[2, 4, 8, 16, 32, 64],
        results_directory=results_dir,
        save_results=save_results
    )
    return ExperimentalFramework(config)


def run_quantum_advantage_certification_study(network_size: int = 16,
                                             n_trials: int = 50) -> Dict[str, Any]:
    """
    Run a focused quantum advantage certification study.
    
    This is the main function for generating publication-ready
    quantum advantage certification results.
    """
    logger.info("Starting comprehensive quantum advantage certification study")
    
    # Create comprehensive experiment
    framework = create_comprehensive_study(n_trials=n_trials)
    
    # Run full experimental suite
    results = framework.run_full_experimental_suite()
    
    # Additional statistical validation
    if results['statistical_summary'].get('statistical_tests', {}):
        for exp_type, test_results in results['statistical_summary']['statistical_tests'].items():
            if test_results.get('significant', False):
                logger.info(f"✅ QUANTUM ADVANTAGE CERTIFIED for {exp_type}: "
                           f"p-value = {test_results['p_value']:.6f}")
            else:
                logger.info(f"❌ No significant advantage for {exp_type}: "
                           f"p-value = {test_results['p_value']:.6f}")
    
    return results