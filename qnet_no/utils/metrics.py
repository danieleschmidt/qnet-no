"""Comprehensive metrics collection and monitoring for QNet-NO."""

import time
import threading
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import jax
import jax.numpy as jnp
import psutil
import logging
from datetime import datetime, timedelta

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QuantumMetrics:
    """Quantum-specific performance metrics."""
    circuit_fidelity: float = 0.0
    entanglement_quality: float = 0.0
    schmidt_rank: int = 0
    gate_count: int = 0
    circuit_depth: int = 0
    quantum_volume: float = 0.0
    coherence_time: float = 0.0


@dataclass 
class TrainingMetrics:
    """Training performance metrics."""
    epoch: int = 0
    loss: float = float('inf')
    accuracy: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    throughput: float = 0.0  # samples/second
    time_per_epoch: float = 0.0
    gradient_norm: float = 0.0
    convergence_rate: float = 0.0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_total: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)


@dataclass
class DistributedMetrics:
    """Distributed computing metrics."""
    active_nodes: int = 0
    task_queue_size: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    network_latency: float = 0.0
    load_balance_efficiency: float = 0.0


class MetricsCollector:
    """Comprehensive metrics collection system for QNet-NO."""
    
    def __init__(self, enable_prometheus: bool = True, collection_interval: float = 10.0):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.collection_interval = collection_interval
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
        self.running = False
        self.collector_thread = None
        
        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics()
        
        # Current metrics snapshots
        self.current_quantum_metrics = QuantumMetrics()
        self.current_training_metrics = TrainingMetrics()
        self.current_system_metrics = SystemMetrics()
        self.current_distributed_metrics = DistributedMetrics()
        
        # Performance tracking
        self.operation_timers = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.custom_metrics = defaultdict(float)
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not self.enable_prometheus:
            return
            
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Quantum metrics
        self.prom_circuit_fidelity = Gauge(
            'qnet_no_circuit_fidelity', 
            'Quantum circuit fidelity',
            registry=self.registry
        )
        self.prom_entanglement_quality = Gauge(
            'qnet_no_entanglement_quality',
            'Entanglement quality between quantum nodes',  
            registry=self.registry
        )
        self.prom_schmidt_rank = Gauge(
            'qnet_no_schmidt_rank',
            'Schmidt rank of quantum states',
            registry=self.registry
        )
        
        # Training metrics
        self.prom_training_loss = Gauge(
            'qnet_no_training_loss',
            'Training loss value',
            registry=self.registry
        )
        self.prom_training_accuracy = Gauge(
            'qnet_no_training_accuracy', 
            'Training accuracy',
            registry=self.registry
        )
        self.prom_throughput = Gauge(
            'qnet_no_throughput_samples_per_second',
            'Training throughput in samples per second',
            registry=self.registry
        )
        
        # System metrics
        self.prom_memory_usage = Gauge(
            'qnet_no_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        self.prom_gpu_utilization = Gauge(
            'qnet_no_gpu_utilization_ratio',
            'GPU utilization ratio',
            registry=self.registry
        )
        
        # Operation timing
        self.prom_operation_duration = Histogram(
            'qnet_no_operation_duration_seconds',
            'Duration of various operations',
            ['operation_type'],
            registry=self.registry
        )
        
        # Error counting
        self.prom_errors = Counter(
            'qnet_no_errors_total',
            'Total number of errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.prom_cache_hits = Counter(
            'qnet_no_cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )
        self.prom_cache_misses = Counter(
            'qnet_no_cache_misses_total', 
            'Total cache misses',
            registry=self.registry
        )
        
        # Distributed metrics
        self.prom_active_nodes = Gauge(
            'qnet_no_active_nodes',
            'Number of active distributed nodes',
            registry=self.registry
        )
        self.prom_task_queue_size = Gauge(
            'qnet_no_task_queue_size',
            'Size of distributed task queue',
            registry=self.registry
        )
        
    def start_collection(self):
        """Start automatic metrics collection."""
        if self.running:
            return
            
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        logger.info("Metrics collection started")
        
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
        
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._update_prometheus_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # GPU metrics (if available)
            gpu_usage = 0.0
            gpu_memory = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUsed / gpu.memoryTotal * 100
            except ImportError:
                pass
            
            with self.lock:
                self.current_system_metrics = SystemMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    memory_total=memory.total,
                    gpu_usage=gpu_usage,
                    gpu_memory=gpu_memory,
                    disk_usage=disk.percent,
                    network_io=network_stats
                )
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics."""
        if not self.enable_prometheus:
            return
            
        try:
            with self.lock:
                # Update quantum metrics
                self.prom_circuit_fidelity.set(self.current_quantum_metrics.circuit_fidelity)
                self.prom_entanglement_quality.set(self.current_quantum_metrics.entanglement_quality)
                self.prom_schmidt_rank.set(self.current_quantum_metrics.schmidt_rank)
                
                # Update training metrics
                self.prom_training_loss.set(self.current_training_metrics.loss)
                self.prom_training_accuracy.set(self.current_training_metrics.accuracy)
                self.prom_throughput.set(self.current_training_metrics.throughput)
                
                # Update system metrics
                self.prom_memory_usage.set(self.current_system_metrics.memory_total * 
                                         self.current_system_metrics.memory_usage / 100)
                self.prom_gpu_utilization.set(self.current_system_metrics.gpu_usage / 100)
                
                # Update distributed metrics
                self.prom_active_nodes.set(self.current_distributed_metrics.active_nodes)
                self.prom_task_queue_size.set(self.current_distributed_metrics.task_queue_size)
                
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
            
    def record_quantum_metrics(self, **kwargs):
        """Record quantum-specific metrics."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.current_quantum_metrics, key):
                    setattr(self.current_quantum_metrics, key, value)
                    self.metrics_history[f'quantum_{key}'].append({
                        'timestamp': time.time(),
                        'value': value
                    })
                    
    def record_training_metrics(self, **kwargs):
        """Record training metrics."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.current_training_metrics, key):
                    setattr(self.current_training_metrics, key, value)
                    self.metrics_history[f'training_{key}'].append({
                        'timestamp': time.time(),
                        'value': value
                    })
                    
    def record_distributed_metrics(self, **kwargs):
        """Record distributed computing metrics."""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.current_distributed_metrics, key):
                    setattr(self.current_distributed_metrics, key, value)
                    self.metrics_history[f'distributed_{key}'].append({
                        'timestamp': time.time(),
                        'value': value
                    })
                    
    def record_operation_time(self, operation: str, duration: float):
        """Record operation timing."""
        with self.lock:
            self.operation_timers[operation].append(duration)
            if len(self.operation_timers[operation]) > 100:
                self.operation_timers[operation].pop(0)
                
        if self.enable_prometheus:
            self.prom_operation_duration.labels(operation_type=operation).observe(duration)
            
    def record_error(self, error_type: str):
        """Record error occurrence."""
        with self.lock:
            self.error_counts[error_type] += 1
            
        if self.enable_prometheus:
            self.prom_errors.labels(error_type=error_type).inc()
            
    def record_cache_hit(self):
        """Record cache hit."""
        if self.enable_prometheus:
            self.prom_cache_hits.inc()
            
    def record_cache_miss(self):
        """Record cache miss."""
        if self.enable_prometheus:
            self.prom_cache_misses.inc()
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self.lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'quantum_metrics': {
                    'circuit_fidelity': self.current_quantum_metrics.circuit_fidelity,
                    'entanglement_quality': self.current_quantum_metrics.entanglement_quality,
                    'schmidt_rank': self.current_quantum_metrics.schmidt_rank,
                    'gate_count': self.current_quantum_metrics.gate_count,
                    'circuit_depth': self.current_quantum_metrics.circuit_depth,
                    'quantum_volume': self.current_quantum_metrics.quantum_volume
                },
                'training_metrics': {
                    'epoch': self.current_training_metrics.epoch,
                    'loss': self.current_training_metrics.loss,
                    'accuracy': self.current_training_metrics.accuracy,
                    'throughput': self.current_training_metrics.throughput,
                    'learning_rate': self.current_training_metrics.learning_rate
                },
                'system_metrics': {
                    'cpu_usage': self.current_system_metrics.cpu_usage,
                    'memory_usage': self.current_system_metrics.memory_usage,
                    'gpu_usage': self.current_system_metrics.gpu_usage,
                    'disk_usage': self.current_system_metrics.disk_usage
                },
                'distributed_metrics': {
                    'active_nodes': self.current_distributed_metrics.active_nodes,
                    'task_queue_size': self.current_distributed_metrics.task_queue_size,
                    'completed_tasks': self.current_distributed_metrics.completed_tasks,
                    'failed_tasks': self.current_distributed_metrics.failed_tasks
                },
                'operation_timings': {
                    op: {
                        'count': len(times),
                        'mean': np.mean(times) if times else 0,
                        'std': np.std(times) if times else 0,
                        'min': np.min(times) if times else 0,
                        'max': np.max(times) if times else 0
                    }
                    for op, times in self.operation_timers.items()
                },
                'error_counts': dict(self.error_counts),
                'custom_metrics': dict(self.custom_metrics)
            }
            
        return summary
        
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        if not self.enable_prometheus:
            return ""
            
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode('utf-8')
        
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file."""
        import json
        
        summary = self.get_metrics_summary()
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2)
            elif format.lower() == 'csv':
                import pandas as pd
                # Flatten the nested structure for CSV
                flat_data = []
                timestamp = summary['timestamp']
                
                for category, metrics in summary.items():
                    if isinstance(metrics, dict) and category != 'timestamp':
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                flat_data.append({
                                    'timestamp': timestamp,
                                    'category': category,
                                    'metric': metric_name,
                                    'value': value
                                })
                
                df = pd.DataFrame(flat_data)
                df.to_csv(filepath, index=False)
                
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            
    def get_performance_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        report = {
            'report_period': f"Last {time_window_hours} hours",
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_metrics_summary()
        }
        
        # Analyze trends
        with self.lock:
            trends = {}
            for metric_name, history in self.metrics_history.items():
                recent_data = [
                    entry for entry in history 
                    if entry['timestamp'] > cutoff_time
                ]
                
                if len(recent_data) >= 2:
                    values = [entry['value'] for entry in recent_data]
                    trends[metric_name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'trend': 'increasing' if values[-1] > values[0] else 'decreasing',
                        'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
                    
        report['trends'] = trends
        
        # Performance insights
        insights = []
        
        # Check for performance issues
        if self.current_system_metrics.memory_usage > 90:
            insights.append("High memory usage detected - consider increasing memory or optimizing algorithms")
            
        if self.current_system_metrics.gpu_usage < 30:
            insights.append("Low GPU utilization - workload might not be GPU-optimized")
            
        if self.current_quantum_metrics.circuit_fidelity < 0.9:
            insights.append("Low quantum circuit fidelity - check for noise or calibration issues")
            
        if self.current_training_metrics.loss == float('inf'):
            insights.append("Training loss is infinite - model may not be properly initialized")
            
        # Check error rates
        total_operations = sum(len(times) for times in self.operation_timers.values())
        total_errors = sum(self.error_counts.values())
        if total_operations > 0 and (total_errors / total_operations) > 0.05:
            insights.append(f"High error rate detected: {total_errors/total_operations:.2%}")
            
        report['insights'] = insights
        
        return report


# Global metrics collector instance
_global_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
        _global_collector.start_collection()
    return _global_collector


def record_quantum_operation(func):
    """Decorator to record quantum operation metrics."""
    def wrapper(*args, **kwargs):
        collector = get_metrics_collector()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            collector.record_operation_time(func.__name__, duration)
            return result
        except Exception as e:
            collector.record_error(type(e).__name__)
            raise
            
    return wrapper


def record_training_step(func):
    """Decorator to record training step metrics."""
    def wrapper(*args, **kwargs):
        collector = get_metrics_collector()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract metrics from result if available
            if isinstance(result, dict):
                if 'loss' in result:
                    collector.record_training_metrics(loss=result['loss'])
                if 'accuracy' in result:
                    collector.record_training_metrics(accuracy=result['accuracy'])
                    
            collector.record_operation_time(f"training_{func.__name__}", duration)
            return result
        except Exception as e:
            collector.record_error(f"training_{type(e).__name__}")
            raise
            
    return wrapper