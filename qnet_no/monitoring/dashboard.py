"""Real-time monitoring dashboard for QNet-NO."""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from ..utils.metrics import get_metrics_collector, MetricsCollector

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Real-time monitoring dashboard for QNet-NO quantum neural operators."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.refresh_interval = 5  # seconds
        self.max_data_points = 100
        
    def create_quantum_metrics_chart(self) -> go.Figure:
        """Create quantum-specific metrics visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Circuit Fidelity', 'Entanglement Quality', 
                          'Schmidt Rank', 'Quantum Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Get historical data
        history = self.metrics_collector.metrics_history
        
        # Circuit Fidelity
        fidelity_data = history.get('quantum_circuit_fidelity', [])
        if fidelity_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in fidelity_data[-50:]]
            values = [d['value'] for d in fidelity_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Fidelity', 
                          line=dict(color='blue')),
                row=1, col=1
            )
            
        # Entanglement Quality
        entanglement_data = history.get('quantum_entanglement_quality', [])
        if entanglement_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in entanglement_data[-50:]]
            values = [d['value'] for d in entanglement_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Entanglement', 
                          line=dict(color='green')),
                row=1, col=2
            )
            
        # Schmidt Rank
        schmidt_data = history.get('quantum_schmidt_rank', [])
        if schmidt_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in schmidt_data[-50:]]
            values = [d['value'] for d in schmidt_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Schmidt Rank', 
                          line=dict(color='red')),
                row=2, col=1
            )
            
        # Quantum Volume
        volume_data = history.get('quantum_quantum_volume', [])
        if volume_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in volume_data[-50:]]
            values = [d['value'] for d in volume_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Quantum Volume', 
                          line=dict(color='purple')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Quantum Metrics Overview",
            height=600,
            showlegend=False
        )
        
        return fig
        
    def create_training_metrics_chart(self) -> go.Figure:
        """Create training metrics visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Accuracy', 'Throughput', 'Learning Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        history = self.metrics_collector.metrics_history
        
        # Training Loss
        loss_data = history.get('training_loss', [])
        if loss_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in loss_data[-50:]]
            values = [d['value'] for d in loss_data[-50:] if d['value'] != float('inf')]
            
            if values:  # Only plot if we have valid loss values
                fig.add_trace(
                    go.Scatter(x=timestamps[:len(values)], y=values, name='Loss', 
                              line=dict(color='red')),
                    row=1, col=1
                )
            
        # Accuracy
        acc_data = history.get('training_accuracy', [])
        if acc_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in acc_data[-50:]]
            values = [d['value'] for d in acc_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Accuracy', 
                          line=dict(color='green')),
                row=1, col=2
            )
            
        # Throughput
        throughput_data = history.get('training_throughput', [])
        if throughput_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in throughput_data[-50:]]
            values = [d['value'] for d in throughput_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Throughput', 
                          line=dict(color='blue')),
                row=2, col=1
            )
            
        # Learning Rate
        lr_data = history.get('training_learning_rate', [])
        if lr_data:
            timestamps = [datetime.fromtimestamp(d['timestamp']) for d in lr_data[-50:]]
            values = [d['value'] for d in lr_data[-50:]]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Learning Rate', 
                          line=dict(color='orange')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Training Metrics Overview",
            height=600,
            showlegend=False
        )
        
        return fig
        
    def create_system_metrics_chart(self) -> go.Figure:
        """Create system resource metrics visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'GPU Usage (%)', 'Disk Usage (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Get current system metrics
        system_metrics = self.metrics_collector.current_system_metrics
        
        # Create gauge-like visualizations
        cpu_value = system_metrics.cpu_usage
        memory_value = system_metrics.memory_usage  
        gpu_value = system_metrics.gpu_usage
        disk_value = system_metrics.disk_usage
        
        # CPU Usage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=cpu_value,
                domain={'x': [0, 0.5], 'y': [0.5, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=memory_value,
                domain={'x': [0.5, 1], 'y': [0.5, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=2
        )
        
        # GPU Usage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=gpu_value,
                domain={'x': [0, 0.5], 'y': [0, 0.5]},
                title={'text': "GPU Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=2, col=1
        )
        
        # Disk Usage
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=disk_value,
                domain={'x': [0.5, 1], 'y': [0, 0.5]},
                title={'text': "Disk Usage (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="System Resource Usage",
            height=600
        )
        
        return fig
        
    def create_distributed_metrics_chart(self) -> go.Figure:
        """Create distributed computing metrics visualization."""
        distributed_metrics = self.metrics_collector.current_distributed_metrics
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Active Nodes', 'Task Queue Size', 'Task Success Rate', 'Average Task Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Active Nodes
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=distributed_metrics.active_nodes,
                title={'text': "Active Nodes"},
                number={'font': {'size': 40}},
                domain={'x': [0, 0.5], 'y': [0.5, 1]}
            ),
            row=1, col=1
        )
        
        # Task Queue Size
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=distributed_metrics.task_queue_size,
                title={'text': "Queued Tasks"},
                number={'font': {'size': 40}},
                domain={'x': [0.5, 1], 'y': [0.5, 1]}
            ),
            row=1, col=2
        )
        
        # Task Success Rate
        total_tasks = distributed_metrics.completed_tasks + distributed_metrics.failed_tasks
        success_rate = (distributed_metrics.completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=success_rate,
                title={'text': "Success Rate (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "red"},
                        {'range': [80, 95], 'color': "yellow"},
                        {'range': [95, 100], 'color': "green"}
                    ]
                },
                domain={'x': [0, 0.5], 'y': [0, 0.5]}
            ),
            row=2, col=1
        )
        
        # Average Task Time
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=distributed_metrics.average_task_time,
                title={'text': "Avg Task Time (s)"},
                number={'font': {'size': 40}, 'suffix': "s"},
                domain={'x': [0.5, 1], 'y': [0, 0.5]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Distributed Computing Metrics",
            height=600
        )
        
        return fig
        
    def create_error_analysis_chart(self) -> go.Figure:
        """Create error analysis visualization."""
        error_counts = self.metrics_collector.error_counts
        
        if not error_counts:
            # Return empty chart if no errors
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No errors recorded",
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title="Error Analysis",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Create pie chart of error types
        error_types = list(error_counts.keys())
        error_values = list(error_counts.values())
        
        fig = go.Figure(data=[
            go.Pie(
                labels=error_types,
                values=error_values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Error Distribution by Type",
            height=400
        )
        
        return fig
        
    def create_performance_overview(self) -> Dict[str, Any]:
        """Create performance overview summary."""
        summary = self.metrics_collector.get_metrics_summary()
        
        # Calculate derived metrics
        current_time = datetime.now()
        quantum_health = "Excellent"
        training_status = "Normal"
        system_status = "Healthy"
        
        # Assess quantum health
        fidelity = summary['quantum_metrics']['circuit_fidelity']
        entanglement = summary['quantum_metrics']['entanglement_quality']
        
        if fidelity < 0.8 or entanglement < 0.6:
            quantum_health = "Poor"
        elif fidelity < 0.9 or entanglement < 0.8:
            quantum_health = "Fair"
        elif fidelity < 0.95 or entanglement < 0.9:
            quantum_health = "Good"
        
        # Assess training status
        loss = summary['training_metrics']['loss']
        if loss == float('inf'):
            training_status = "Not Started"
        elif loss > 1.0:
            training_status = "Unstable"
        elif loss > 0.1:
            training_status = "Converging"
        else:
            training_status = "Converged"
        
        # Assess system status
        cpu = summary['system_metrics']['cpu_usage']
        memory = summary['system_metrics']['memory_usage']
        
        if cpu > 90 or memory > 95:
            system_status = "Critical"
        elif cpu > 80 or memory > 85:
            system_status = "Warning"
        
        return {
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'quantum_health': quantum_health,
            'training_status': training_status,
            'system_status': system_status,
            'active_nodes': summary['distributed_metrics']['active_nodes'],
            'total_errors': sum(self.metrics_collector.error_counts.values()),
            'uptime': "N/A"  # Would need to track start time
        }


def run_streamlit_dashboard():
    """Run the Streamlit dashboard application."""
    if not STREAMLIT_AVAILABLE:
        logger.error("Streamlit not available. Install with: pip install streamlit plotly")
        return
        
    st.set_page_config(
        page_title="QNet-NO Monitoring Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 QNet-NO Quantum Neural Operator Monitoring")
    
    # Initialize dashboard
    dashboard = MonitoringDashboard()
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 60, 5)
    
    if st.sidebar.button("Manual Refresh"):
        st.rerun()
        
    # Performance Overview
    st.subheader("📊 Performance Overview")
    overview = dashboard.create_performance_overview()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Quantum Health", overview['quantum_health'])
    with col2:
        st.metric("Training Status", overview['training_status'])
    with col3:
        st.metric("System Status", overview['system_status'])
    with col4:
        st.metric("Active Nodes", overview['active_nodes'])
    
    # Quantum Metrics
    st.subheader("⚛️ Quantum Metrics")
    quantum_chart = dashboard.create_quantum_metrics_chart()
    st.plotly_chart(quantum_chart, use_container_width=True)
    
    # Training Metrics
    st.subheader("🎯 Training Metrics")
    training_chart = dashboard.create_training_metrics_chart()
    st.plotly_chart(training_chart, use_container_width=True)
    
    # System Metrics
    st.subheader("💻 System Resources")
    system_chart = dashboard.create_system_metrics_chart()
    st.plotly_chart(system_chart, use_container_width=True)
    
    # Distributed Metrics
    st.subheader("🌐 Distributed Computing")
    distributed_chart = dashboard.create_distributed_metrics_chart()
    st.plotly_chart(distributed_chart, use_container_width=True)
    
    # Error Analysis
    st.subheader("⚠️ Error Analysis")
    error_chart = dashboard.create_error_analysis_chart()
    st.plotly_chart(error_chart, use_container_width=True)
    
    # Raw Metrics (expandable)
    with st.expander("📋 Raw Metrics Data"):
        summary = dashboard.metrics_collector.get_metrics_summary()
        st.json(summary)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    run_streamlit_dashboard()