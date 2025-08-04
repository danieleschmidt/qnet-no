#!/usr/bin/env python3
"""
QNet-NO Generation 3 Scaling Demonstration

This example demonstrates the full capabilities of Generation 3 (Make it Scale):
- Performance optimization with memory pooling and computation caching
- Distributed computing across multiple quantum nodes
- Auto-scaling based on network load and performance metrics
- Comprehensive monitoring and metrics collection
- Production deployment readiness

Usage:
    python examples/scaling_demonstration.py --mode [demo|benchmark|monitor]
"""

import argparse
import time
import threading
from typing import Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import logging
from pathlib import Path

# QNet-NO imports
from qnet_no.operators.quantum_fno import QuantumFourierNeuralOperator
from qnet_no.networks.photonic_network import PhotonicNetwork, QuantumNode
from qnet_no.utils.metrics import get_metrics_collector
from qnet_no.utils.performance import MemoryPool, ComputationCache
from qnet_no.utils.distributed import create_distributed_cluster
from qnet_no.monitoring.dashboard import MonitoringDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_pde_data(n_samples: int = 1000, spatial_size: int = 64) -> Dict[str, jnp.ndarray]:
    """Create synthetic PDE data for demonstration."""
    logger.info(f"Generating synthetic PDE data: {n_samples} samples, {spatial_size}x{spatial_size} spatial grid")
    
    # Generate spatial coordinates
    x = jnp.linspace(0, 2*jnp.pi, spatial_size)
    y = jnp.linspace(0, 2*jnp.pi, spatial_size)
    X, Y = jnp.meshgrid(x, y)
    
    # Generate synthetic input fields (initial conditions)
    inputs = []
    targets = []
    
    for i in range(n_samples):
        # Random wave parameters
        k1, k2 = np.random.uniform(1, 3, 2)
        a1, a2 = np.random.uniform(0.5, 2.0, 2)
        phi1, phi2 = np.random.uniform(0, 2*np.pi, 2)
        
        # Initial condition: superposition of sine waves
        u0 = a1 * jnp.sin(k1 * X + phi1) + a2 * jnp.cos(k2 * Y + phi2)
        
        # Target: time evolution (simplified Burger's equation solution)
        t = 0.1
        u_evolved = u0 * jnp.exp(-0.1 * t * (k1**2 + k2**2))
        
        inputs.append(u0[..., None])  # Add channel dimension
        targets.append(u_evolved[..., None])
    
    return {
        'inputs': jnp.array(inputs),
        'targets': jnp.array(targets)
    }


def setup_quantum_network(n_nodes: int = 4) -> PhotonicNetwork:
    """Set up quantum photonic network for demonstration."""
    logger.info(f"Setting up quantum network with {n_nodes} nodes")
    
    network = PhotonicNetwork(nodes=n_nodes, topology="ring")
    
    # Add quantum nodes with varying capabilities
    for i in range(n_nodes):
        node = QuantumNode(
            node_id=i,
            n_qubits=16 + i * 2,  # Varying qubit counts
            fidelity=0.95 + i * 0.01,  # Varying fidelities
            connectivity=min(4, n_nodes - 1)
        )
        network.add_node(node)
    
    # Establish entanglement channels
    network.establish_entanglement_channels()
    
    logger.info(f"Quantum network initialized with {len(network.quantum_nodes)} nodes")
    return network


def demonstrate_basic_functionality():
    """Demonstrate basic QNet-NO functionality."""
    logger.info("=== Demonstrating Basic Functionality ===")
    
    # Create network and data
    network = setup_quantum_network(n_nodes=2)
    data = create_synthetic_pde_data(n_samples=100, spatial_size=32)
    
    # Initialize quantum FNO
    qfno = QuantumFourierNeuralOperator(
        modes=8,
        width=32,
        schmidt_rank=4,
        n_layers=2
    )
    
    # Train model
    logger.info("Training quantum FNO...")
    results = qfno.fit(
        train_data=data,
        network=network,
        epochs=5,
        lr=1e-3,
        batch_size=16
    )
    
    logger.info(f"Training completed. Final loss: {results['losses'][-1]:.6f}")
    
    # Make predictions
    test_data = create_synthetic_pde_data(n_samples=20, spatial_size=32)
    predictions = qfno.predict({'inputs': test_data['inputs']}, network)
    
    logger.info(f"Predictions completed. Output shape: {predictions.shape}")
    return results


def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    logger.info("=== Demonstrating Performance Optimization ===")
    
    network = setup_quantum_network(n_nodes=3)
    
    # Create larger dataset to showcase optimization
    data = create_synthetic_pde_data(n_samples=500, spatial_size=64)
    
    qfno = QuantumFourierNeuralOperator(
        modes=16,
        width=64,
        schmidt_rank=8,
        n_layers=4
    )
    
    # Demonstrate memory pooling
    logger.info("Memory pool statistics before training:")
    memory_stats = qfno.memory_pool.get_statistics()
    logger.info(f"Pool size: {memory_stats['total_size_gb']:.2f} GB, Utilization: {memory_stats['utilization']:.2%}")
    
    # Train with caching enabled
    logger.info("Training with performance optimization enabled...")
    start_time = time.time()
    
    results = qfno.fit(
        train_data=data,
        network=network,
        epochs=10,
        lr=1e-3,
        batch_size=32
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Show performance statistics
    perf_stats = results.get('performance_stats', {})
    cache_stats = results.get('cache_stats', {})
    memory_stats = results.get('memory_stats', {})
    
    logger.info("Performance Statistics:")
    logger.info(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
    logger.info(f"  Memory utilization: {memory_stats.get('utilization', 0):.2%}")
    logger.info(f"  Total operations: {perf_stats.get('total_operations', 0)}")
    
    return results


def demonstrate_distributed_computing():
    """Demonstrate distributed computing capabilities."""
    logger.info("=== Demonstrating Distributed Computing ===")
    
    # Set up larger network
    network = setup_quantum_network(n_nodes=6)
    
    # Create node configurations for distributed computing
    node_configs = [
        {'host': 'localhost', 'port': 8000 + i, 'capabilities': ['gpu', 'quantum', 'fft', 'tensor']}
        for i in range(4)
    ]
    
    # Initialize QNet-NO with distributed capabilities
    qfno = QuantumFourierNeuralOperator(
        modes=32,
        width=128,
        schmidt_rank=16,
        n_layers=6
    )
    
    # Enable distributed computing
    logger.info("Enabling distributed computing...")
    qfno.enable_distributed_computing(node_configs)
    
    # Create large dataset to benefit from distribution
    data = create_synthetic_pde_data(n_samples=1000, spatial_size=128)
    
    logger.info("Training with distributed computing...")
    start_time = time.time()
    
    results = qfno.fit(
        train_data=data,
        network=network,
        epochs=8,
        lr=1e-3,
        batch_size=64
    )
    
    training_time = time.time() - start_time
    logger.info(f"Distributed training completed in {training_time:.2f} seconds")
    
    # Show distributed statistics
    scaling_stats = qfno.get_scaling_statistics()
    scheduler_stats = scaling_stats.get('scheduler_stats', {})
    
    logger.info("Distributed Computing Statistics:")
    logger.info(f"  Active nodes: {scheduler_stats.get('completed_tasks', 0)}")
    logger.info(f"  Completed tasks: {scheduler_stats.get('completed_tasks', 0)}")
    logger.info(f"  Failed tasks: {scheduler_stats.get('failed_tasks', 0)}")
    logger.info(f"  Task queue size: {scheduler_stats.get('queued_tasks', 0)}")
    
    return results


def demonstrate_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    logger.info("=== Demonstrating Auto-Scaling ===")
    
    network = setup_quantum_network(n_nodes=4)
    
    qfno = QuantumFourierNeuralOperator(
        modes=24,
        width=96,
        schmidt_rank=12,
        n_layers=5
    )
    
    # Enable auto-scaling
    logger.info("Enabling auto-scaling...")
    qfno.enable_auto_scaling(network, target_utilization=0.75)
    
    # Simulate varying workloads
    datasets = [
        create_synthetic_pde_data(n_samples=200, spatial_size=32),
        create_synthetic_pde_data(n_samples=500, spatial_size=64),
        create_synthetic_pde_data(n_samples=800, spatial_size=96),
        create_synthetic_pde_data(n_samples=300, spatial_size=48)
    ]
    
    results_list = []
    
    for i, data in enumerate(datasets):
        logger.info(f"Training on dataset {i+1}/4 (size: {len(data['inputs'])})")
        
        # Adjust batch size based on current performance
        if i > 0:
            last_result = results_list[-1]
            last_loss = last_result['losses'][-1] if last_result['losses'] else 1.0
            memory_usage = qfno.memory_pool.get_statistics()['utilization'] * qfno.memory_pool.max_pool_size
            throughput = len(data['inputs']) / 60  # Rough estimate
            
            optimal_batch_size = qfno.auto_scale_batch_size(
                current_loss=float(last_loss),
                memory_usage=memory_usage,
                throughput=throughput
            )
            logger.info(f"Auto-scaled batch size: {optimal_batch_size}")
        else:
            optimal_batch_size = 32
        
        results = qfno.fit(
            train_data=data,
            network=network,
            epochs=5,
            lr=1e-3,
            batch_size=optimal_batch_size
        )
        results_list.append(results)
        
        # Show current scaling statistics
        scaling_stats = qfno.get_scaling_statistics()
        logger.info(f"Memory utilization: {scaling_stats['memory_stats']['utilization']:.2%}")
        logger.info(f"Cache hit rate: {scaling_stats['cache_stats']['hit_rate']:.2%}")
        
        # Short pause to observe auto-scaling behavior
        time.sleep(2)
    
    logger.info("Auto-scaling demonstration completed")
    return results_list


def run_comprehensive_benchmark():
    """Run comprehensive benchmark of all Generation 3 features."""
    logger.info("=== Running Comprehensive Benchmark ===")
    
    # Metrics collection
    metrics_collector = get_metrics_collector()
    
    benchmark_results = {
        'basic_functionality': None,
        'performance_optimization': None,
        'distributed_computing': None,
        'auto_scaling': None,
        'total_time': 0
    }
    
    start_time = time.time()
    
    try:
        # Run all demonstrations
        benchmark_results['basic_functionality'] = demonstrate_basic_functionality()
        benchmark_results['performance_optimization'] = demonstrate_performance_optimization()
        benchmark_results['distributed_computing'] = demonstrate_distributed_computing()
        benchmark_results['auto_scaling'] = demonstrate_auto_scaling()
        
        benchmark_results['total_time'] = time.time() - start_time
        
        # Generate comprehensive performance report
        performance_report = metrics_collector.get_performance_report(time_window_hours=1)
        
        logger.info("=== Benchmark Summary ===")
        logger.info(f"Total benchmark time: {benchmark_results['total_time']:.2f} seconds")
        logger.info("Performance Insights:")
        for insight in performance_report['insights']:
            logger.info(f"  • {insight}")
        
        # Export results
        export_path = Path("benchmark_results.json")
        metrics_collector.export_metrics(str(export_path), format='json')
        logger.info(f"Detailed results exported to {export_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
    
    return benchmark_results


def run_monitoring_demo():
    """Run monitoring dashboard demonstration."""
    logger.info("=== Starting Monitoring Dashboard Demo ===")
    
    try:
        from qnet_no.monitoring.dashboard import run_streamlit_dashboard
        
        # Start background activity to generate metrics
        def background_activity():
            network = setup_quantum_network(n_nodes=3)
            qfno = QuantumFourierNeuralOperator(modes=16, width=64, schmidt_rank=8, n_layers=3)
            
            while True:
                data = create_synthetic_pde_data(n_samples=50, spatial_size=32)
                qfno.fit(train_data=data, network=network, epochs=3, batch_size=16)
                time.sleep(30)
        
        # Start background activity
        activity_thread = threading.Thread(target=background_activity, daemon=True)
        activity_thread.start()
        
        logger.info("Starting Streamlit dashboard...")
        logger.info("Open your browser to http://localhost:8501 to view the dashboard")
        
        # Run dashboard
        run_streamlit_dashboard()
        
    except ImportError:
        logger.error("Streamlit not available. Install with: pip install streamlit plotly")
        logger.info("Running basic monitoring demo instead...")
        
        # Alternative: show metrics in console
        metrics_collector = get_metrics_collector()
        
        for i in range(10):
            summary = metrics_collector.get_metrics_summary()
            logger.info(f"Metrics snapshot {i+1}:")
            logger.info(f"  System CPU: {summary['system_metrics']['cpu_usage']:.1f}%")
            logger.info(f"  System Memory: {summary['system_metrics']['memory_usage']:.1f}%")
            logger.info(f"  Quantum Fidelity: {summary['quantum_metrics']['circuit_fidelity']:.3f}")
            time.sleep(5)


def main():
    """Main entry point for scaling demonstration."""
    parser = argparse.ArgumentParser(
        description="QNet-NO Generation 3 Scaling Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scaling_demonstration.py --mode demo          # Run all demonstrations
  python scaling_demonstration.py --mode benchmark     # Run comprehensive benchmark
  python scaling_demonstration.py --mode monitor       # Start monitoring dashboard
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'benchmark', 'monitor'],
        default='demo',
        help='Demonstration mode to run'
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 QNet-NO Generation 3: Make it Scale")
    logger.info("="*60)
    
    try:
        if args.mode == 'demo':
            # Run individual demonstrations
            demonstrate_basic_functionality()
            time.sleep(2)
            demonstrate_performance_optimization()
            time.sleep(2)
            demonstrate_distributed_computing()
            time.sleep(2)
            demonstrate_auto_scaling()
            
        elif args.mode == 'benchmark':
            run_comprehensive_benchmark()
            
        elif args.mode == 'monitor':
            run_monitoring_demo()
            
        logger.info("✅ Demonstration completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()