#!/usr/bin/env python3
"""
QNet-NO Command Line Interface

Provides command-line tools for quantum neural operator experiments.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import qnet_no as qno


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="qnet-no",
        description="Quantum-Network Neural Operator Library CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qnet-no demo                    # Run quickstart demo
  qnet-no benchmark --suite all  # Run comprehensive benchmarks
  qnet-no train --data navier_stokes --epochs 100
  qnet-no network --nodes 8 --topology grid --visualize
        """
    )
    
    parser.add_argument(
        "--version", action="version", 
        version=f"QNet-NO {qno.__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quickstart demonstration")
    demo_parser.add_argument(
        "--advanced", action="store_true",
        help="Run advanced examples"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument(
        "--suite", choices=["all", "linear", "nonlinear", "multiscale"], 
        default="all", help="Benchmark suite to run"
    )
    benchmark_parser.add_argument(
        "--resolution", type=int, default=64,
        help="Spatial resolution for benchmarks"
    )
    benchmark_parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of samples per dataset"
    )
    benchmark_parser.add_argument(
        "--output", type=str, help="Output file for results"
    )
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train quantum neural operators")
    train_parser.add_argument(
        "--data", choices=["navier_stokes", "heat", "wave", "burgers", "darcy", "maxwell"],
        required=True, help="PDE dataset to use"
    )
    train_parser.add_argument(
        "--model", choices=["qfno", "deeponet", "hybrid"], 
        default="qfno", help="Model architecture"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    train_parser.add_argument(
        "--nodes", type=int, default=4, help="Number of quantum nodes"
    )
    train_parser.add_argument(
        "--save", type=str, help="Path to save trained model"
    )
    
    # Network command
    network_parser = subparsers.add_parser("network", help="Quantum network tools")
    network_parser.add_argument(
        "--nodes", type=int, default=4, help="Number of quantum nodes"
    )
    network_parser.add_argument(
        "--topology", choices=["complete", "ring", "star", "grid", "random"],
        default="complete", help="Network topology"
    )
    network_parser.add_argument(
        "--protocol", choices=["nv_center", "photonic", "ion_trap"],
        default="nv_center", help="Entanglement protocol"
    )
    network_parser.add_argument(
        "--fidelity", type=float, default=0.85, help="Fidelity threshold"
    )
    network_parser.add_argument(
        "--visualize", action="store_true", help="Visualize network topology"
    )
    network_parser.add_argument(
        "--stats", action="store_true", help="Display network statistics"
    )
    
    # Backend command
    backend_parser = subparsers.add_parser("backend", help="Quantum backend tools")
    backend_parser.add_argument(
        "--type", choices=["simulator", "photonic", "nv_center"],
        default="simulator", help="Backend type"
    )
    backend_parser.add_argument(
        "--test", action="store_true", help="Test backend connection"
    )
    backend_parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark backend performance"
    )
    
    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management")
    dataset_parser.add_argument(
        "--generate", choices=["navier_stokes", "heat", "wave", "burgers", "synthetic"],
        help="Generate dataset"
    )
    dataset_parser.add_argument(
        "--resolution", type=int, default=64, help="Spatial resolution"
    )
    dataset_parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples"
    )
    dataset_parser.add_argument(
        "--output", type=str, help="Output directory"
    )
    
    return parser


def run_demo(args):
    """Run demonstration examples."""
    print("🚀 Running QNet-NO Demo")
    
    if args.advanced:
        print("Running advanced examples...")
        try:
            from examples.advanced_examples import main
            main()
        except ImportError:
            print("⚠️ Advanced examples not found. Run from repository root.")
            return 1
    else:
        print("Running quickstart demo...")
        try:
            from examples.quickstart_demo import main
            main()
        except ImportError:
            print("⚠️ Quickstart demo not found. Run from repository root.")
            return 1
    
    return 0


def run_benchmark(args):
    """Run performance benchmarks."""
    print(f"🏆 Running {args.suite} benchmarks")
    print(f"Resolution: {args.resolution}, Samples: {args.samples}")
    
    try:
        # Create benchmark suite
        if args.suite == "all":
            from qnet_no.datasets import create_benchmark_suite
            suite = create_benchmark_suite(args.resolution, args.samples // 6)  # Divide by 6 datasets
        else:
            print(f"⚠️ Specific suite '{args.suite}' not implemented yet")
            return 1
        
        # Run benchmarks
        results = {}
        
        for name, dataset in suite.datasets.items():
            print(f"\n📊 Benchmarking {name}...")
            
            # Create test network and model
            network = qno.PhotonicNetwork(nodes=4)
            model = qno.QuantumFourierNeuralOperator(modes=16, width=64)
            
            # Quick training
            try:
                train_results = model.fit(
                    dataset.train, network, epochs=5, batch_size=16
                )
                
                predictions = model.predict(dataset.test, network)
                mse = float(jnp.mean((predictions - dataset.test['targets']) ** 2))
                
                results[name] = {
                    'mse': mse,
                    'final_loss': float(train_results['losses'][-1]),
                    'equation': dataset.metadata.get('equation', 'unknown')
                }
                
                print(f"   ✅ MSE: {mse:.6f}")
                
            except Exception as e:
                print(f"   ⚠️ Failed: {e}")
                results[name] = {'error': str(e)}
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
        # Summary
        print(f"\n📈 Benchmark Summary:")
        successful = [name for name, result in results.items() if 'mse' in result]
        print(f"   Successful: {len(successful)}/{len(results)} datasets")
        
        if successful:
            avg_mse = sum(results[name]['mse'] for name in successful) / len(successful)
            print(f"   Average MSE: {avg_mse:.6f}")
        
    except Exception as e:
        print(f"⚠️ Benchmark failed: {e}")
        return 1
    
    return 0


def run_training(args):
    """Run model training."""
    print(f"🚀 Training {args.model} on {args.data}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    try:
        # Load dataset
        if args.data == "navier_stokes":
            data = qno.datasets.load_navier_stokes(n_samples=500)
        elif args.data == "heat":
            data = qno.datasets.load_heat_equation(n_samples=500)
        elif args.data == "wave":
            data = qno.datasets.load_wave_equation(n_samples=500)
        elif args.data == "burgers":
            data = qno.datasets.load_burgers_equation(n_samples=500)
        else:
            print(f"⚠️ Dataset {args.data} not implemented")
            return 1
        
        print(f"✅ Loaded {args.data}: {data.metadata['n_train']} train, {data.metadata['n_test']} test")
        
        # Create network
        network = qno.PhotonicNetwork(nodes=args.nodes)
        print(f"✅ Created quantum network: {args.nodes} nodes")
        
        # Create model
        if args.model == "qfno":
            model = qno.QuantumFourierNeuralOperator()
        elif args.model == "deeponet":
            model = qno.QuantumDeepONet()
        elif args.model == "hybrid":
            model = qno.HybridNeuralOperator()
        
        print(f"✅ Created {args.model} model")
        
        # Train
        print("🏃 Starting training...")
        results = model.fit(
            data.train, network,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
        
        print(f"✅ Training completed! Final loss: {results['losses'][-1]:.6f}")
        
        # Evaluate
        predictions = model.predict(data.test, network)
        mse = jnp.mean((predictions - data.test['targets']) ** 2)
        print(f"📊 Test MSE: {mse:.6f}")
        
        # Save model
        if args.save:
            # In real implementation would save model parameters
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_info = {
                'model_type': args.model,
                'dataset': args.data,
                'final_loss': float(results['losses'][-1]),
                'test_mse': float(mse),
                'epochs': args.epochs
            }
            
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"💾 Model info saved to {save_path.with_suffix('.json')}")
        
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
        return 1
    
    return 0


def run_network(args):
    """Run network management tools."""
    print(f"🌐 Quantum Network: {args.nodes} nodes, {args.topology} topology")
    
    try:
        # Create network
        network = qno.PhotonicNetwork(
            nodes=args.nodes,
            topology=args.topology,
            entanglement_protocol=args.protocol,
            fidelity_threshold=args.fidelity
        )
        
        print(f"✅ Network created successfully")
        
        if args.stats:
            stats = network.get_network_stats()
            print(f"\n📊 Network Statistics:")
            print(f"   Nodes: {stats['num_nodes']}")
            print(f"   Links: {stats['num_links']}")
            print(f"   Total qubits: {stats['total_qubits']}")
            print(f"   Avg node fidelity: {stats['avg_node_fidelity']:.3f}")
            print(f"   Avg link fidelity: {stats['avg_link_fidelity']:.3f}")
            print(f"   Protocols: {stats['protocols']}")
        
        if args.visualize:
            print("🎨 Creating network visualization...")
            try:
                network.visualize_network()
                print("✅ Network visualization displayed")
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
        
    except Exception as e:
        print(f"⚠️ Network creation failed: {e}")
        return 1
    
    return 0


def run_backend(args):
    """Run backend management tools."""
    print(f"🖥️ Quantum Backend: {args.type}")
    
    try:
        # Create backend
        if args.type == "simulator":
            backend = qno.backends.SimulatorBackend(n_qubits=8)
        elif args.type == "photonic":
            backend = qno.backends.PhotonicBackend(n_modes=8)
        elif args.type == "nv_center":
            backend = qno.backends.NVCenterBackend(n_qubits=6)
        
        if args.test:
            print("🔌 Testing backend connection...")
            if backend.connect():
                print("✅ Backend connected successfully")
                
                props = backend.get_backend_properties()
                print(f"   Type: {props['backend_type']}")
                print(f"   Qubits/Modes: {props.get('n_qubits', props.get('n_modes'))}")
                
                backend.disconnect()
            else:
                print("❌ Backend connection failed")
                return 1
        
        if args.benchmark:
            print("⚡ Running backend benchmark...")
            # Simple benchmark circuit
            from qnet_no.backends.base_backend import QuantumCircuit
            
            circuit = QuantumCircuit(
                gates=[
                    {"gate": "h", "qubit": 0},
                    {"gate": "cnot", "control": 0, "target": 1}
                ],
                n_qubits=8,
                measurements=[0, 1]
            )
            
            if backend.connect():
                result = backend.execute_circuit(circuit, shots=1000)
                print(f"✅ Benchmark completed")
                print(f"   Execution time: {result.execution_time:.3f} μs")
                print(f"   Fidelity: {result.fidelity:.3f}")
                print(f"   Measurement outcomes: {len(result.measurement_counts)}")
                
                backend.disconnect()
            else:
                print("❌ Could not connect for benchmark")
                return 1
        
    except Exception as e:
        print(f"⚠️ Backend operation failed: {e}")
        return 1
    
    return 0


def run_dataset(args):
    """Run dataset management tools."""
    if args.generate:
        print(f"📊 Generating {args.generate} dataset")
        print(f"Resolution: {args.resolution}, Samples: {args.samples}")
        
        try:
            if args.generate == "navier_stokes":
                data = qno.datasets.load_navier_stokes(
                    resolution=args.resolution,
                    n_samples=args.samples
                )
            elif args.generate == "heat":
                data = qno.datasets.load_heat_equation(
                    resolution=args.resolution,
                    n_samples=args.samples
                )
            elif args.generate == "synthetic":
                data = qno.datasets.generate_synthetic_pde_data(
                    "linear", args.resolution, args.samples
                )
            else:
                print(f"⚠️ Dataset type {args.generate} not implemented")
                return 1
            
            print(f"✅ Dataset generated successfully")
            print(f"   Train samples: {data.metadata['n_train']}")
            print(f"   Test samples: {data.metadata['n_test']}")
            
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save metadata
                with open(output_path / "metadata.json", 'w') as f:
                    json.dump(data.metadata, f, indent=2)
                
                print(f"💾 Dataset metadata saved to {output_path}")
            
        except Exception as e:
            print(f"⚠️ Dataset generation failed: {e}")
            return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Import JAX numpy here to avoid startup overhead
    global jnp
    import jax.numpy as jnp
    
    # Route to appropriate handler
    handlers = {
        'demo': run_demo,
        'benchmark': run_benchmark,
        'train': run_training,
        'network': run_network,
        'backend': run_backend,
        'dataset': run_dataset,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"⚠️ Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())