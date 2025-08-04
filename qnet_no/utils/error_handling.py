"""Comprehensive error handling and recovery mechanisms for QNet-NO."""

import functools
import logging
import traceback
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
import jax.numpy as jnp

# Set up logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QuantumError(Exception):
    """Base exception for quantum-specific errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class NetworkError(QuantumError):
    """Errors related to quantum network operations."""
    pass


class BackendError(QuantumError):
    """Errors related to quantum backend operations."""
    pass


class OperatorError(QuantumError):
    """Errors related to quantum neural operators."""
    pass


class DataError(QuantumError):
    """Errors related to data processing and validation."""
    pass


class TrainingError(QuantumError):
    """Errors related to model training."""
    pass


class ErrorRecovery:
    """Error recovery strategies and fallback mechanisms."""
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_retries: int = 3, 
                          base_delay: float = 1.0, backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,)) -> Any:
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            backoff_factor: Multiplier for delay after each retry
            exceptions: Tuple of exceptions to catch and retry
            
        Returns:
            Result of successful function execution
            
        Raises:
            Last encountered exception if all retries fail
        """
        last_exception = None
        delay = base_delay
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries} after {delay:.1f}s delay")
                    time.sleep(delay)
                
                result = func()
                
                if attempt > 0:
                    logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except exceptions as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    delay *= backoff_factor
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    @staticmethod
    def fallback_to_classical(quantum_func: Callable, classical_func: Callable,
                             context: str = "operation") -> Any:
        """
        Attempt quantum operation with classical fallback.
        
        Args:
            quantum_func: Primary quantum function to try
            classical_func: Fallback classical function
            context: Description of operation for logging
            
        Returns:
            Result from quantum or classical function
        """
        try:
            logger.debug(f"Attempting quantum {context}")
            result = quantum_func()
            logger.info(f"Quantum {context} completed successfully")
            return result
            
        except Exception as e:
            logger.warning(f"Quantum {context} failed: {e}")
            logger.info(f"Falling back to classical {context}")
            
            try:
                result = classical_func()
                logger.info(f"Classical {context} completed successfully")
                return result
                
            except Exception as classical_error:
                logger.error(f"Both quantum and classical {context} failed")
                raise QuantumError(
                    f"Both quantum and classical {context} failed. "
                    f"Quantum error: {e}. Classical error: {classical_error}",
                    severity=ErrorSeverity.HIGH,
                    context={"quantum_error": str(e), "classical_error": str(classical_error)}
                )
    
    @staticmethod
    def graceful_degradation(primary_func: Callable, degraded_func: Callable,
                           performance_threshold: float = 0.8,
                           context: str = "operation") -> Any:
        """
        Gracefully degrade performance if primary operation fails.
        
        Args:
            primary_func: High-performance function to try first
            degraded_func: Lower-performance but more reliable function
            performance_threshold: Threshold for switching to degraded mode
            context: Description for logging
            
        Returns:
            Result from primary or degraded function
        """
        try:
            start_time = time.time()
            result = primary_func()
            execution_time = time.time() - start_time
            
            # Check if performance is acceptable
            if hasattr(result, '__len__') and len(result) > 0:
                # If result has performance metrics, check them
                performance = getattr(result, 'performance', 1.0)
                if performance < performance_threshold:
                    logger.warning(f"Performance degraded in {context}: {performance:.3f}")
                    logger.info(f"Switching to degraded mode for {context}")
                    return degraded_func()
            
            logger.debug(f"Primary {context} completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            logger.warning(f"Primary {context} failed: {e}")
            logger.info(f"Using degraded {context}")
            
            try:
                result = degraded_func()
                logger.info(f"Degraded {context} completed successfully")
                return result
                
            except Exception as degraded_error:
                raise QuantumError(
                    f"Both primary and degraded {context} failed. "
                    f"Primary error: {e}. Degraded error: {degraded_error}",
                    severity=ErrorSeverity.HIGH
                )


def error_boundary(error_type: Type[QuantumError] = QuantumError,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  reraise: bool = True, default_return: Any = None):
    """
    Decorator for creating error boundaries around functions.
    
    Args:
        error_type: Type of quantum error to wrap exceptions in
        severity: Severity level for wrapped errors
        reraise: Whether to reraise the wrapped error
        default_return: Default value to return if error occurs and reraise=False
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except QuantumError:
                # Already a quantum error, just reraise
                raise
                
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args_types': [type(arg).__name__ for arg in args],
                    'kwargs_keys': list(kwargs.keys()),
                    'traceback': traceback.format_exc()
                }
                
                wrapped_error = error_type(
                    f"Error in {func.__name__}: {e}",
                    severity=severity,
                    context=context
                )
                
                logger.error(f"Error boundary caught exception in {func.__name__}: {e}")
                
                if reraise:
                    raise wrapped_error
                else:
                    logger.warning(f"Returning default value for {func.__name__}")
                    return default_return
        
        return wrapper
    return decorator


def validate_inputs(validation_func: Callable):
    """
    Decorator to validate function inputs before execution.
    
    Args:
        validation_func: Function that validates inputs and returns ValidationResult
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run validation
            try:
                validation_result = validation_func(*args, **kwargs)
                
                if not validation_result.is_valid:
                    error_msg = f"Input validation failed for {func.__name__}: {validation_result.errors}"
                    raise DataError(error_msg, severity=ErrorSeverity.HIGH,
                                  context={'validation_errors': validation_result.errors})
                
                # Log warnings if any
                if validation_result.warnings:
                    for warning, level in validation_result.warnings:
                        if level == ErrorSeverity.HIGH:
                            logger.warning(f"{func.__name__}: {warning}")
                        else:
                            logger.info(f"{func.__name__}: {warning}")
                
            except Exception as e:
                raise DataError(f"Validation failed for {func.__name__}: {e}",
                              severity=ErrorSeverity.HIGH)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class QuantumResourceMonitor:
    """Monitor quantum resources and detect potential issues."""
    
    def __init__(self):
        self.memory_usage = []
        self.computation_times = []
        self.error_counts = {}
        
    def record_memory_usage(self, usage_mb: float):
        """Record memory usage measurement."""
        self.memory_usage.append(usage_mb)
        
        # Check for memory leaks
        if len(self.memory_usage) > 10:
            recent_avg = sum(self.memory_usage[-5:]) / 5
            older_avg = sum(self.memory_usage[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.5:  # 50% increase
                logger.warning(f"Potential memory leak detected: {recent_avg:.1f}MB vs {older_avg:.1f}MB")
    
    def record_computation_time(self, operation: str, time_seconds: float):
        """Record computation time for performance monitoring."""
        if operation not in self.computation_times:
            self.computation_times[operation] = []
        
        self.computation_times[operation].append(time_seconds)
        
        # Check for performance degradation
        if len(self.computation_times[operation]) > 5:
            recent_avg = sum(self.computation_times[operation][-3:]) / 3
            baseline = sum(self.computation_times[operation][:3]) / 3
            
            if recent_avg > baseline * 2.0:  # 2x slower
                logger.warning(f"Performance degradation in {operation}: "
                             f"{recent_avg:.3f}s vs {baseline:.3f}s baseline")
    
    def record_error(self, error_type: str):
        """Record error occurrence for trend analysis."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Alert on high error rates
        if self.error_counts[error_type] > 10:
            logger.error(f"High error rate for {error_type}: {self.error_counts[error_type]} occurrences")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        status = {"healthy": True, "issues": []}
        
        # Check memory trends
        if len(self.memory_usage) > 5:
            current_memory = self.memory_usage[-1]
            if current_memory > 1000:  # 1GB
                status["issues"].append(f"High memory usage: {current_memory:.1f}MB")
                status["healthy"] = False
        
        # Check error rates
        for error_type, count in self.error_counts.items():
            if count > 5:
                status["issues"].append(f"High {error_type} error rate: {count}")
                status["healthy"] = False
        
        # Check performance
        for operation, times in self.computation_times.items():
            if len(times) > 3:
                avg_time = sum(times[-3:]) / 3
                if avg_time > 60:  # More than 1 minute
                    status["issues"].append(f"Slow {operation}: {avg_time:.1f}s average")
        
        return status


# Global resource monitor
resource_monitor = QuantumResourceMonitor()


def monitor_resources(operation_name: str):
    """Decorator to monitor resource usage of operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful completion
                execution_time = time.time() - start_time
                resource_monitor.record_computation_time(operation_name, execution_time)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                resource_monitor.record_error(error_type)
                raise
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for quantum operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise QuantumError(
                    "Circuit breaker is OPEN - operation not attempted",
                    severity=ErrorSeverity.HIGH
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - update state
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker moving to CLOSED state")
            elif self.state == "CLOSED":
                self.failure_count = 0  # Reset failure count on success
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opening after {self.failure_count} failures")
            
            raise e


def safe_quantum_operation(circuit_breaker: Optional[CircuitBreaker] = None,
                          max_retries: int = 2, fallback_classical: bool = True):
    """
    Comprehensive decorator for safe quantum operations.
    
    Combines circuit breaker, retry logic, and classical fallback.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract classical fallback if provided
            classical_fallback = kwargs.pop('classical_fallback', None)
            
            def quantum_operation():
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            # Try quantum operation with retries
            try:
                return ErrorRecovery.retry_with_backoff(
                    quantum_operation,
                    max_retries=max_retries,
                    exceptions=(QuantumError, NetworkError, BackendError)
                )
                
            except Exception as e:
                if fallback_classical and classical_fallback:
                    logger.warning(f"Quantum operation failed, using classical fallback: {e}")
                    return classical_fallback(*args, **kwargs)
                else:
                    raise QuantumError(
                        f"Quantum operation failed: {e}",
                        severity=ErrorSeverity.HIGH,
                        context={'original_error': str(e)}
                    )
        
        return wrapper
    return decorator


class ErrorAggregator:
    """Aggregate and analyze error patterns."""
    
    def __init__(self):
        self.errors = []
        self.error_patterns = {}
    
    def add_error(self, error: QuantumError):
        """Add error to aggregation."""
        self.errors.append(error)
        
        # Track patterns
        error_signature = f"{type(error).__name__}:{error.severity.value}"
        if error_signature not in self.error_patterns:
            self.error_patterns[error_signature] = []
        
        self.error_patterns[error_signature].append(error.timestamp)
    
    def get_error_summary(self, time_window: float = 3600) -> Dict[str, Any]:
        """Get summary of errors in time window (default: last hour)."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_errors = [e for e in self.errors if e.timestamp > cutoff_time]
        
        summary = {
            "total_errors": len(recent_errors),
            "by_type": {},
            "by_severity": {},
            "most_common": None,
            "recommendations": []
        }
        
        # Count by type and severity
        for error in recent_errors:
            error_type = type(error).__name__
            severity = error.severity.value
            
            summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        # Find most common error
        if summary["by_type"]:
            most_common = max(summary["by_type"].items(), key=lambda x: x[1])
            summary["most_common"] = f"{most_common[0]} ({most_common[1]} occurrences)"
        
        # Generate recommendations
        if summary["by_severity"].get("critical", 0) > 0:
            summary["recommendations"].append("Critical errors detected - immediate attention required")
        
        if summary["by_severity"].get("high", 0) > 5:
            summary["recommendations"].append("High number of high-severity errors - review system configuration")
        
        if len(recent_errors) > 50:
            summary["recommendations"].append("Very high error rate - consider circuit breaker or system restart")
        
        return summary


# Global error aggregator
error_aggregator = ErrorAggregator()


def report_error(error: QuantumError):
    """Report error to global aggregator and logging system."""
    error_aggregator.add_error(error)
    
    # Log based on severity
    if error.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"CRITICAL ERROR: {error}")
    elif error.severity == ErrorSeverity.HIGH:
        logger.error(f"HIGH SEVERITY: {error}")
    elif error.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"MEDIUM SEVERITY: {error}")
    else:
        logger.info(f"LOW SEVERITY: {error}")
    
    # Log context if available
    if error.context:
        logger.debug(f"Error context: {error.context}")