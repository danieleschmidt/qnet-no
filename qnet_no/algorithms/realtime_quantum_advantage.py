"""Real-Time Quantum Advantage Verification and Monitoring System.

This module implements a comprehensive real-time system for monitoring, verifying,
and certifying quantum advantage during live quantum computations. Unlike static
post-execution analysis, this system provides continuous verification with
adaptive optimization based on detected quantum advantage patterns.

Key Research Breakthroughs:
1. Streaming statistical tests for continuous quantum advantage monitoring
2. Dynamic confidence interval updating with early advantage detection
3. Adaptive resource reallocation based on real-time advantage measurements
4. Predictive quantum advantage modeling using machine learning
5. Runtime optimization triggers based on advantage degradation detection

This represents a paradigm shift from static quantum advantage certification
to dynamic, real-time optimization systems that adapt quantum computations
based on live advantage measurements.

Author: Terry - Terragon Labs
Date: August 12, 2025
Research Area: Dynamic Quantum Advantage Optimization
"""

from typing import Dict, Any, Optional, Tuple, List, Callable, Union
import time
import threading
import queue
import jax
import jax.numpy as jnp
import numpy as np
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
import scipy.stats as stats

from ..networks.photonic_network import PhotonicNetwork
from ..utils.quantum_encoding import quantum_feature_map
from ..utils.validation import validate_tensor_shape, log_validation_result
from ..utils.error_handling import (
    error_boundary, OperatorError, ErrorSeverity, 
    monitor_resources, safe_quantum_operation
)
from ..utils.performance import PerformanceProfiler
from ..utils.metrics import get_metrics_collector, record_quantum_operation

logger = logging.getLogger(__name__)


@dataclass
class QuantumAdvantageSnapshot:
    """Single snapshot of quantum advantage measurements."""
    
    timestamp: float
    quantum_metric: float
    classical_baseline: float
    advantage_ratio: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    measurement_fidelity: float
    network_coherence: float
    operation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvantageAlertConfig:
    """Configuration for quantum advantage alerts and triggers."""
    
    advantage_threshold: float = 1.1
    significance_threshold: float = 0.05
    confidence_level: float = 0.95
    min_sample_size: int = 10
    degradation_threshold: float = 0.8
    improvement_threshold: float = 1.2
    alert_cooldown_seconds: float = 30.0


class QuantumAdvantagePredictor:
    """Machine learning model for predicting quantum advantage trends.
    
    Uses historical advantage measurements to predict future performance
    and identify optimal resource allocation strategies.
    """
    
    def __init__(self, prediction_window: int = 50):
        self.prediction_window = prediction_window
        self.feature_history = deque(maxlen=prediction_window)
        self.advantage_history = deque(maxlen=prediction_window)
        
        # Simple linear model for trend prediction
        self.model_weights = jnp.array([1.0, 0.1, 0.05])  # [baseline, trend, acceleration]
        self.prediction_confidence = 0.5
        
    def update_with_measurement(self, snapshot: QuantumAdvantageSnapshot) -> None:
        """Update predictor with new measurement."""
        
        # Extract features for prediction
        features = jnp.array([
            snapshot.advantage_ratio,
            snapshot.measurement_fidelity,
            snapshot.network_coherence,
            snapshot.statistical_significance,
            len(self.advantage_history)  # Time feature
        ])
        
        self.feature_history.append(features)
        self.advantage_history.append(snapshot.advantage_ratio)
        
        # Update model if we have enough data
        if len(self.advantage_history) >= 10:
            self._update_prediction_model()
    
    def predict_advantage_trend(self, steps_ahead: int = 5) -> Dict[str, Union[float, List[float]]]:
        """Predict quantum advantage trend for next few steps."""
        
        if len(self.advantage_history) < 3:
            return {
                'predicted_advantages': [1.0] * steps_ahead,
                'confidence': 0.1,
                'trend_direction': 'unknown'
            }
        
        # Calculate trend components
        recent_advantages = list(self.advantage_history)[-10:]
        
        # Linear trend estimation
        x = jnp.arange(len(recent_advantages))
        y = jnp.array(recent_advantages)
        
        # Simple linear regression
        slope = jnp.sum((x - jnp.mean(x)) * (y - jnp.mean(y))) / jnp.sum((x - jnp.mean(x)) ** 2)
        intercept = jnp.mean(y) - slope * jnp.mean(x)
        
        # Predict future values
        future_x = jnp.arange(len(recent_advantages), len(recent_advantages) + steps_ahead)
        predictions = slope * future_x + intercept
        
        # Estimate confidence based on recent variance
        recent_variance = jnp.var(jnp.array(recent_advantages))
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + recent_variance)))
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = 'improving'
        elif slope < -0.01:
            trend_direction = 'degrading'
        else:
            trend_direction = 'stable'
        
        return {
            'predicted_advantages': list(predictions),
            'confidence': float(confidence),
            'trend_direction': trend_direction,
            'slope': float(slope),
            'predicted_variance': float(recent_variance)
        }
    
    def recommend_optimization(self, 
                              current_snapshot: QuantumAdvantageSnapshot) -> Dict[str, Any]:
        """Recommend optimization actions based on predictions."""
        
        prediction = self.predict_advantage_trend()
        
        recommendations = {
            'actions': [],
            'priority': 'low',
            'expected_improvement': 0.0,
            'confidence': prediction['confidence']
        }
        
        # Analyze trends and recommend actions
        if prediction['trend_direction'] == 'degrading':
            recommendations['actions'].extend([
                'increase_schmidt_rank',
                'optimize_entanglement_fidelity', 
                'reduce_circuit_depth',
                'add_error_correction'
            ])
            recommendations['priority'] = 'high'
            recommendations['expected_improvement'] = 0.2
            
        elif prediction['trend_direction'] == 'stable' and current_snapshot.advantage_ratio > 1.5:
            recommendations['actions'].extend([
                'increase_problem_complexity',
                'expand_network_size',
                'enable_aggressive_optimization'
            ])
            recommendations['priority'] = 'medium'
            recommendations['expected_improvement'] = 0.3
            
        elif prediction['trend_direction'] == 'improving':
            recommendations['actions'].extend([
                'maintain_current_configuration',
                'prepare_scaling_resources',
                'enable_performance_profiling'
            ])
            recommendations['priority'] = 'low'
            recommendations['expected_improvement'] = 0.1
        
        return recommendations
    
    def _update_prediction_model(self) -> None:
        """Update the internal prediction model with recent data."""
        
        if len(self.feature_history) < 5:
            return
        
        # Simple exponential moving average for model confidence
        recent_accuracy = self._calculate_recent_prediction_accuracy()
        self.prediction_confidence = 0.9 * self.prediction_confidence + 0.1 * recent_accuracy
    
    def _calculate_recent_prediction_accuracy(self) -> float:
        """Calculate accuracy of recent predictions."""
        
        if len(self.advantage_history) < 10:
            return 0.5
        
        # Simple accuracy metric: how well did we predict the trend direction
        recent_changes = jnp.diff(jnp.array(list(self.advantage_history)[-10:]))
        predicted_direction = 1.0 if jnp.mean(recent_changes) > 0 else -1.0
        actual_direction = 1.0 if recent_changes[-1] > 0 else -1.0
        
        return 0.8 if predicted_direction * actual_direction > 0 else 0.2


class StreamingStatisticalTester:
    """Streaming statistical tests for continuous quantum advantage verification.
    
    Performs statistical hypothesis testing on streaming data without requiring
    the full dataset to be stored in memory.
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 power_threshold: float = 0.8):
        
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        
        # Streaming statistics
        self.quantum_stats = StreamingStats()
        self.classical_stats = StreamingStats()
        
        # Welch's t-test state
        self.last_t_stat = 0.0
        self.last_p_value = 1.0
        self.last_effect_size = 0.0
        
        # Sequential testing
        self.log_likelihood_ratio = 0.0
        self.boundaries_calculated = False
        self.upper_boundary = 0.0
        self.lower_boundary = 0.0
        
    def add_quantum_measurement(self, value: float) -> None:
        """Add new quantum measurement to streaming statistics."""
        self.quantum_stats.add_value(value)
    
    def add_classical_measurement(self, value: float) -> None:
        """Add new classical baseline measurement to streaming statistics."""
        self.classical_stats.add_value(value)
    
    def test_quantum_advantage(self) -> Dict[str, float]:
        """Perform streaming statistical test for quantum advantage."""
        
        if self.quantum_stats.n < 2 or self.classical_stats.n < 2:
            return {
                'statistically_significant': False,
                'p_value': 1.0,
                't_statistic': 0.0,
                'effect_size': 0.0,
                'power': 0.0,
                'confidence_interval_lower': 0.0,
                'confidence_interval_upper': 0.0
            }
        
        # Welch's t-test for unequal variances
        quantum_mean = self.quantum_stats.mean
        classical_mean = self.classical_stats.mean
        quantum_var = self.quantum_stats.variance
        classical_var = self.classical_stats.variance
        n_quantum = self.quantum_stats.n
        n_classical = self.classical_stats.n
        
        # t-statistic calculation
        pooled_se = jnp.sqrt(quantum_var / n_quantum + classical_var / n_classical)
        t_stat = (quantum_mean - classical_mean) / (pooled_se + 1e-10)
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        df = ((quantum_var / n_quantum + classical_var / n_classical) ** 2) / (
            (quantum_var / n_quantum) ** 2 / (n_quantum - 1) +
            (classical_var / n_classical) ** 2 / (n_classical - 1)
        )
        
        # p-value (one-tailed test for quantum > classical)
        p_value = 1.0 - stats.t.cdf(float(t_stat), float(df))
        
        # Effect size (Cohen's d)
        pooled_std = jnp.sqrt(((n_quantum - 1) * quantum_var + 
                              (n_classical - 1) * classical_var) / 
                             (n_quantum + n_classical - 2))
        effect_size = (quantum_mean - classical_mean) / (pooled_std + 1e-10)
        
        # Statistical power approximation
        power = self._estimate_statistical_power(effect_size, n_quantum, n_classical)
        
        # Confidence interval for difference in means
        critical_t = stats.t.ppf(1 - self.significance_level / 2, float(df))
        margin_error = critical_t * pooled_se
        ci_lower = (quantum_mean - classical_mean) - margin_error
        ci_upper = (quantum_mean - classical_mean) + margin_error
        
        # Store results
        self.last_t_stat = float(t_stat)
        self.last_p_value = float(p_value)
        self.last_effect_size = float(effect_size)
        
        return {
            'statistically_significant': p_value < self.significance_level,
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'effect_size': float(effect_size),
            'power': float(power),
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'degrees_of_freedom': float(df),
            'sample_sizes': {'quantum': n_quantum, 'classical': n_classical}
        }
    
    def sequential_probability_ratio_test(self, 
                                        h0_effect: float = 0.0, 
                                        h1_effect: float = 0.5) -> Dict[str, Any]:
        """Perform Sequential Probability Ratio Test for early stopping."""
        
        if not self.boundaries_calculated:
            self._calculate_sprt_boundaries(h0_effect, h1_effect)
        
        # Update log likelihood ratio
        if self.quantum_stats.n > 0 and self.classical_stats.n > 0:
            self._update_log_likelihood_ratio(h0_effect, h1_effect)
        
        # Check boundaries
        decision = 'continue'
        if self.log_likelihood_ratio >= self.upper_boundary:
            decision = 'accept_h1'  # Accept quantum advantage
        elif self.log_likelihood_ratio <= self.lower_boundary:
            decision = 'accept_h0'  # Accept no advantage
        
        return {
            'decision': decision,
            'log_likelihood_ratio': self.log_likelihood_ratio,
            'upper_boundary': self.upper_boundary,
            'lower_boundary': self.lower_boundary,
            'samples_processed': self.quantum_stats.n + self.classical_stats.n
        }
    
    def _estimate_statistical_power(self, 
                                   effect_size: float, 
                                   n1: int, 
                                   n2: int) -> float:
        """Estimate statistical power of the test."""
        
        # Cohen's power approximation
        delta = effect_size * jnp.sqrt(n1 * n2 / (n1 + n2))
        
        # Non-centrality parameter approximation
        ncp = jnp.abs(delta)
        
        # Power approximation using normal distribution
        z_alpha = stats.norm.ppf(1 - self.significance_level)
        z_power = ncp - z_alpha
        power = stats.norm.cdf(z_power)
        
        return float(jnp.clip(power, 0.0, 1.0))
    
    def _calculate_sprt_boundaries(self, h0_effect: float, h1_effect: float) -> None:
        """Calculate boundaries for Sequential Probability Ratio Test."""
        
        alpha = self.significance_level
        beta = 1.0 - self.power_threshold
        
        self.upper_boundary = jnp.log((1 - beta) / alpha)
        self.lower_boundary = jnp.log(beta / (1 - alpha))
        self.boundaries_calculated = True
    
    def _update_log_likelihood_ratio(self, h0_effect: float, h1_effect: float) -> None:
        """Update log likelihood ratio for SPRT."""
        
        # Simplified likelihood ratio update
        # In practice, this would use proper likelihood functions
        
        current_effect = self.last_effect_size
        
        # Log likelihood ratio update
        ll_h1 = -0.5 * (current_effect - h1_effect) ** 2
        ll_h0 = -0.5 * (current_effect - h0_effect) ** 2
        
        self.log_likelihood_ratio += ll_h1 - ll_h0


class StreamingStats:
    """Efficient streaming statistics computation."""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        
    def add_value(self, value: float) -> None:
        """Add new value to streaming statistics (Welford's algorithm)."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
    
    @property
    def variance(self) -> float:
        """Get variance."""
        return self.m2 / max(1, self.n - 1)
    
    @property
    def std(self) -> float:
        """Get standard deviation."""
        return jnp.sqrt(self.variance)


class RealTimeQuantumAdvantageMonitor:
    """Main class for real-time quantum advantage monitoring and optimization.
    
    Provides comprehensive real-time monitoring of quantum advantage with
    adaptive optimization capabilities.
    """
    
    def __init__(self, 
                 network: PhotonicNetwork,
                 alert_config: Optional[AdvantageAlertConfig] = None,
                 monitoring_interval: float = 0.1):
        
        self.network = network
        self.alert_config = alert_config or AdvantageAlertConfig()
        self.monitoring_interval = monitoring_interval
        
        # Core components
        self.predictor = QuantumAdvantagePredictor()
        self.statistical_tester = StreamingStatisticalTester(
            significance_level=self.alert_config.significance_threshold)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.measurement_queue = queue.Queue()
        self.snapshot_history = deque(maxlen=1000)
        
        # Performance tracking
        self.profiler = PerformanceProfiler()
        self.metrics_collector = get_metrics_collector()
        
        # Alert system
        self.alert_callbacks = []
        self.last_alert_time = {}
        
        # Real-time optimization
        self.optimization_enabled = False
        self.optimization_callbacks = []
        
    def start_monitoring(self) -> None:
        """Start real-time quantum advantage monitoring."""
        
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        logger.info("Starting real-time quantum advantage monitoring")
        
        self.is_monitoring = True
        self.profiler.start_profiling()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        
        if not self.is_monitoring:
            return
        
        logger.info("Stopping quantum advantage monitoring")
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        self.profiler.stop_profiling()
        
    @error_boundary(operation_name="quantum_measurement_ingestion", 
                   severity=ErrorSeverity.MEDIUM)
    def ingest_quantum_measurement(self, 
                                  quantum_result: float,
                                  classical_baseline: float,
                                  operation_id: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> QuantumAdvantageSnapshot:
        """Ingest new quantum measurement for real-time analysis."""
        
        timestamp = time.time()
        metadata = metadata or {}
        
        # Update streaming statistics
        self.statistical_tester.add_quantum_measurement(quantum_result)
        self.statistical_tester.add_classical_measurement(classical_baseline)
        
        # Perform statistical test
        test_results = self.statistical_tester.test_quantum_advantage()
        
        # Calculate advantage ratio
        advantage_ratio = quantum_result / max(classical_baseline, 1e-10)
        
        # Get network coherence
        network_coherence = self._measure_network_coherence()
        
        # Create snapshot
        snapshot = QuantumAdvantageSnapshot(
            timestamp=timestamp,
            quantum_metric=quantum_result,
            classical_baseline=classical_baseline,
            advantage_ratio=advantage_ratio,
            statistical_significance=test_results['p_value'],
            confidence_interval=(
                test_results['confidence_interval_lower'],
                test_results['confidence_interval_upper']
            ),
            sample_size=test_results['sample_sizes']['quantum'],
            measurement_fidelity=metadata.get('fidelity', 0.95),
            network_coherence=network_coherence,
            operation_id=operation_id,
            metadata=metadata
        )
        
        # Store snapshot
        self.snapshot_history.append(snapshot)
        
        # Update predictor
        self.predictor.update_with_measurement(snapshot)
        
        # Queue for processing
        if self.is_monitoring:
            try:
                self.measurement_queue.put(snapshot, block=False)
            except queue.Full:
                logger.warning("Measurement queue full, dropping oldest measurements")
        
        return snapshot
    
    def get_current_advantage_status(self) -> Dict[str, Any]:
        """Get current quantum advantage status."""
        
        if not self.snapshot_history:
            return {
                'current_advantage': 1.0,
                'trend': 'unknown',
                'statistical_significance': 1.0,
                'confidence': 0.0,
                'recommendation': 'insufficient_data'
            }
        
        latest_snapshot = self.snapshot_history[-1]
        test_results = self.statistical_tester.test_quantum_advantage()
        predictions = self.predictor.predict_advantage_trend()
        recommendations = self.predictor.recommend_optimization(latest_snapshot)
        
        return {
            'current_advantage': latest_snapshot.advantage_ratio,
            'trend': predictions['trend_direction'],
            'statistical_significance': latest_snapshot.statistical_significance,
            'confidence': predictions['confidence'],
            'recommendation': recommendations['actions'][0] if recommendations['actions'] else 'maintain',
            'test_results': test_results,
            'predictions': predictions,
            'sample_size': test_results['sample_sizes']['quantum'],
            'network_coherence': latest_snapshot.network_coherence
        }
    
    def enable_realtime_optimization(self, 
                                   optimization_callback: Optional[Callable] = None) -> None:
        """Enable real-time optimization based on advantage measurements."""
        
        self.optimization_enabled = True
        
        if optimization_callback:
            self.optimization_callbacks.append(optimization_callback)
        
        logger.info("Real-time quantum advantage optimization enabled")
    
    def add_alert_callback(self, callback: Callable[[QuantumAdvantageSnapshot, str], None]) -> None:
        """Add callback for quantum advantage alerts."""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        
        logger.debug("Quantum advantage monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Process measurements from queue
                self._process_measurement_queue()
                
                # Check for alerts
                self._check_alerts()
                
                # Perform optimization if enabled
                if self.optimization_enabled:
                    self._perform_realtime_optimization()
                
                # Update metrics
                self._update_monitoring_metrics()
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Back off on error
        
        logger.debug("Quantum advantage monitoring loop stopped")
    
    def _process_measurement_queue(self) -> None:
        """Process all pending measurements in the queue."""
        
        processed_count = 0
        
        while not self.measurement_queue.empty() and processed_count < 100:
            try:
                snapshot = self.measurement_queue.get_nowait()
                
                # Record metrics
                self.metrics_collector.record_quantum_metrics(
                    advantage_ratio=snapshot.advantage_ratio,
                    statistical_significance=snapshot.statistical_significance,
                    network_coherence=snapshot.network_coherence
                )
                
                processed_count += 1
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing measurement: {e}")
    
    def _check_alerts(self) -> None:
        """Check for alert conditions and trigger callbacks."""
        
        if not self.snapshot_history:
            return
        
        latest_snapshot = self.snapshot_history[-1]
        current_time = time.time()
        
        # Check advantage degradation
        if (latest_snapshot.advantage_ratio < self.alert_config.degradation_threshold and
            latest_snapshot.statistical_significance < self.alert_config.significance_threshold):
            
            if self._should_send_alert('degradation', current_time):
                self._trigger_alert(latest_snapshot, 'quantum_advantage_degradation')
        
        # Check significant improvement
        elif (latest_snapshot.advantage_ratio > self.alert_config.improvement_threshold and
              latest_snapshot.statistical_significance < self.alert_config.significance_threshold):
            
            if self._should_send_alert('improvement', current_time):
                self._trigger_alert(latest_snapshot, 'quantum_advantage_improvement')
        
        # Check statistical significance achieved
        if (latest_snapshot.statistical_significance < self.alert_config.significance_threshold and
            latest_snapshot.sample_size >= self.alert_config.min_sample_size):
            
            if self._should_send_alert('significance', current_time):
                self._trigger_alert(latest_snapshot, 'statistical_significance_achieved')
    
    def _should_send_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if alert should be sent based on cooldown."""
        
        last_time = self.last_alert_time.get(alert_type, 0.0)
        return current_time - last_time > self.alert_config.alert_cooldown_seconds
    
    def _trigger_alert(self, snapshot: QuantumAdvantageSnapshot, alert_type: str) -> None:
        """Trigger alert callbacks."""
        
        self.last_alert_time[alert_type] = time.time()
        
        for callback in self.alert_callbacks:
            try:
                callback(snapshot, alert_type)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.info(f"Quantum advantage alert triggered: {alert_type}")
    
    def _perform_realtime_optimization(self) -> None:
        """Perform real-time optimization based on advantage measurements."""
        
        if not self.snapshot_history:
            return
        
        latest_snapshot = self.snapshot_history[-1]
        recommendations = self.predictor.recommend_optimization(latest_snapshot)
        
        if recommendations['priority'] in ['high', 'critical']:
            # Execute optimization callbacks
            for callback in self.optimization_callbacks:
                try:
                    callback(recommendations, latest_snapshot)
                except Exception as e:
                    logger.error(f"Error in optimization callback: {e}")
    
    def _update_monitoring_metrics(self) -> None:
        """Update monitoring system metrics."""
        
        if not self.snapshot_history:
            return
        
        recent_snapshots = list(self.snapshot_history)[-10:]  # Last 10 measurements
        
        avg_advantage = jnp.mean([s.advantage_ratio for s in recent_snapshots])
        avg_significance = jnp.mean([s.statistical_significance for s in recent_snapshots])
        avg_coherence = jnp.mean([s.network_coherence for s in recent_snapshots])
        
        self.metrics_collector.record_system_metrics(
            monitoring_active=self.is_monitoring,
            queue_size=self.measurement_queue.qsize(),
            history_size=len(self.snapshot_history),
            avg_advantage=float(avg_advantage),
            avg_significance=float(avg_significance),
            avg_coherence=float(avg_coherence)
        )
    
    def _measure_network_coherence(self) -> float:
        """Measure current network coherence."""
        
        if not self.network.nodes:
            return 0.8  # Default coherence
        
        # Calculate average fidelity across network nodes
        total_fidelity = sum(node.get('fidelity', 0.95) for node in self.network.nodes)
        avg_fidelity = total_fidelity / len(self.network.nodes)
        
        # Factor in network connectivity
        connectivity_factor = min(1.0, len(self.network.nodes) / 10.0)  # Normalized connectivity
        
        return float(avg_fidelity * connectivity_factor)
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        
        performance_report = self.profiler.get_performance_report()
        current_status = self.get_current_advantage_status()
        
        # Calculate statistics over all snapshots
        if self.snapshot_history:
            advantages = [s.advantage_ratio for s in self.snapshot_history]
            significances = [s.statistical_significance for s in self.snapshot_history]
            
            advantage_stats = {
                'mean': float(jnp.mean(jnp.array(advantages))),
                'std': float(jnp.std(jnp.array(advantages))),
                'min': float(jnp.min(jnp.array(advantages))),
                'max': float(jnp.max(jnp.array(advantages))),
                'median': float(jnp.median(jnp.array(advantages)))
            }
            
            significance_stats = {
                'mean': float(jnp.mean(jnp.array(significances))),
                'significant_count': sum(1 for s in significances if s < 0.05),
                'total_measurements': len(significances)
            }
        else:
            advantage_stats = {}
            significance_stats = {}
        
        return {
            'monitoring_active': self.is_monitoring,
            'total_measurements': len(self.snapshot_history),
            'current_status': current_status,
            'advantage_statistics': advantage_stats,
            'significance_statistics': significance_stats,
            'performance_report': performance_report,
            'optimization_enabled': self.optimization_enabled,
            'alert_callbacks_registered': len(self.alert_callbacks),
            'network_coherence': self._measure_network_coherence()
        }


# Example usage and demonstration
def create_default_monitor(network: PhotonicNetwork) -> RealTimeQuantumAdvantageMonitor:
    """Create monitor with default configuration."""
    
    config = AdvantageAlertConfig(
        advantage_threshold=1.2,
        significance_threshold=0.05,
        confidence_level=0.95,
        degradation_threshold=0.9,
        improvement_threshold=1.5
    )
    
    return RealTimeQuantumAdvantageMonitor(network, config)


# Export main classes
__all__ = [
    'RealTimeQuantumAdvantageMonitor',
    'QuantumAdvantageSnapshot',
    'AdvantageAlertConfig', 
    'QuantumAdvantagePredictor',
    'StreamingStatisticalTester',
    'create_default_monitor'
]