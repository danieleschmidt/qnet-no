"""Logging configuration for QNet-NO library."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class QuantumFormatter(logging.Formatter):
    """Custom formatter for quantum operations with color support."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    
    def format(self, record):
        # Add quantum-specific fields
        if hasattr(record, 'quantum_operation'):
            quantum_prefix = f"[Q:{record.quantum_operation}] "
        else:
            quantum_prefix = ""
        
        if hasattr(record, 'node_id'):
            node_prefix = f"[Node-{record.node_id}] "
        else:
            node_prefix = ""
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Format level with colors
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level = f"{self.COLORS[level]}{level}{self.RESET}"
        
        # Create formatted message
        formatted_msg = f"{timestamp} | {level:>8} | {record.name:<25} | {quantum_prefix}{node_prefix}{record.getMessage()}"
        
        # Add exception info if present
        if record.exc_info:
            formatted_msg += f"\n{self.formatException(record.exc_info)}"
        
        return formatted_msg


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': getattr(record, 'module', None),
            'function': getattr(record, 'funcName', None),
            'line': getattr(record, 'lineno', None),
        }
        
        # Add quantum-specific fields
        if hasattr(record, 'quantum_operation'):
            log_entry['quantum_operation'] = record.quantum_operation
        
        if hasattr(record, 'node_id'):
            log_entry['node_id'] = record.node_id
        
        if hasattr(record, 'network_size'):
            log_entry['network_size'] = record.network_size
        
        if hasattr(record, 'schmidt_rank'):
            log_entry['schmidt_rank'] = record.schmidt_rank
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, separators=(',', ':'))


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False,
    console_colors: bool = True
) -> None:
    """
    Set up comprehensive logging for QNet-NO.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        json_format: Use JSON format for structured logging
        console_colors: Use colors in console output
    """
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = QuantumFormatter(use_colors=console_colors)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        # Always use plain format for files (no colors)
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(QuantumFormatter(use_colors=False))
        
        root_logger.addHandler(file_handler)
    
    # Set up QNet-NO specific loggers
    qnet_logger = logging.getLogger('qnet_no')
    qnet_logger.setLevel(numeric_level)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"QNet-NO logging initialized: level={level}, file={log_file}, json={json_format}")


def get_quantum_logger(name: str, quantum_operation: Optional[str] = None, 
                      node_id: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with quantum-specific context.
    
    Args:
        name: Logger name
        quantum_operation: Type of quantum operation
        node_id: Quantum node ID
    
    Returns:
        Configured logger with quantum context
    """
    logger = logging.getLogger(name)
    
    # Add quantum context to all log records
    old_factory = logging.getLogRecordFactory()
    
    def quantum_record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if quantum_operation:
            record.quantum_operation = quantum_operation
        if node_id is not None:
            record.node_id = node_id
        return record
    
    logging.setLogRecordFactory(quantum_record_factory)
    
    return logger


class PerformanceLogger:
    """Logger for performance metrics and quantum advantage analysis."""
    
    def __init__(self, name: str = "qnet_no.performance"):
        self.logger = logging.getLogger(name)
        self.metrics = {}
    
    def log_training_metrics(self, epoch: int, loss: float, time_elapsed: float,
                           network_size: int, schmidt_rank: int):
        """Log training performance metrics."""
        self.logger.info(
            f"Training epoch {epoch}: loss={loss:.6f}, time={time_elapsed:.3f}s",
            extra={
                'quantum_operation': 'training',
                'epoch': epoch,
                'loss': loss,
                'time_elapsed': time_elapsed,
                'network_size': network_size,
                'schmidt_rank': schmidt_rank
            }
        )
        
        # Store metrics for analysis
        if 'training' not in self.metrics:
            self.metrics['training'] = []
        
        self.metrics['training'].append({
            'epoch': epoch,
            'loss': loss,
            'time_elapsed': time_elapsed,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_quantum_advantage(self, operation: str, classical_time: float,
                            quantum_time: float, speedup_factor: float):
        """Log quantum advantage metrics."""
        self.logger.info(
            f"Quantum advantage in {operation}: {speedup_factor:.2f}x speedup "
            f"(classical: {classical_time:.3f}s, quantum: {quantum_time:.3f}s)",
            extra={
                'quantum_operation': 'advantage_analysis',
                'operation': operation,
                'classical_time': classical_time,
                'quantum_time': quantum_time,
                'speedup_factor': speedup_factor
            }
        )
    
    def log_network_stats(self, nodes: int, links: int, avg_fidelity: float,
                         protocols: list):
        """Log quantum network statistics."""
        self.logger.info(
            f"Network stats: {nodes} nodes, {links} links, "
            f"avg_fidelity={avg_fidelity:.3f}, protocols={protocols}",
            extra={
                'quantum_operation': 'network_analysis',
                'network_size': nodes,
                'link_count': links,
                'avg_fidelity': avg_fidelity,
                'protocols': protocols
            }
        )
    
    def export_metrics(self, filepath: str):
        """Export collected metrics to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.logger.info(f"Performance metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


class AuditLogger:
    """Logger for security and compliance auditing."""
    
    def __init__(self, name: str = "qnet_no.audit"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
    
    def log_data_access(self, dataset: str, user: str, operation: str):
        """Log data access events."""
        self.logger.info(
            f"Data access: {user} performed {operation} on {dataset}",
            extra={
                'audit_type': 'data_access',
                'dataset': dataset,
                'user': user,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_model_training(self, model_type: str, dataset: str, epochs: int):
        """Log model training events."""
        self.logger.info(
            f"Model training: {model_type} trained on {dataset} for {epochs} epochs",
            extra={
                'audit_type': 'model_training',
                'model_type': model_type,
                'dataset': dataset,
                'epochs': epochs,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_prediction_request(self, model_type: str, input_shape: tuple, 
                              user: Optional[str] = None):
        """Log prediction requests."""
        self.logger.info(
            f"Prediction request: {model_type} with input shape {input_shape}",
            extra={
                'audit_type': 'prediction',
                'model_type': model_type,
                'input_shape': input_shape,
                'user': user,
                'timestamp': datetime.now().isoformat()
            }
        )


def configure_production_logging():
    """Configure logging for production environment."""
    log_dir = Path(os.getenv('QNET_LOG_DIR', './logs'))
    log_level = os.getenv('QNET_LOG_LEVEL', 'INFO')
    
    setup_logging(
        level=log_level,
        log_file=str(log_dir / 'qnet_no.log'),
        json_format=True,
        console_colors=False  # No colors in production
    )
    
    # Set up performance logging
    perf_logger = PerformanceLogger()
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'performance.log',
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    perf_handler.setFormatter(JSONFormatter())
    perf_logger.logger.addHandler(perf_handler)
    
    # Set up audit logging
    audit_logger = AuditLogger()
    audit_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'audit.log',
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=20  # Keep more audit logs
    )
    audit_handler.setFormatter(JSONFormatter())
    audit_logger.logger.addHandler(audit_handler)


def configure_development_logging():
    """Configure logging for development environment."""
    setup_logging(
        level="DEBUG",
        log_file="./logs/qnet_no_dev.log",
        json_format=False,
        console_colors=True
    )


# Auto-configure based on environment
if os.getenv('QNET_ENV') == 'production':
    configure_production_logging()
elif os.getenv('QNET_ENV') == 'development':
    configure_development_logging()
else:
    # Default development-style logging
    setup_logging(level="INFO", console_colors=True)