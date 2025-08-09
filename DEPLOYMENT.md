# QNet-NO Deployment Guide

**Complete deployment guide for the Hybrid Quantum-Classical Neural Operator framework**

## 🚀 Quick Start Deployment

### Prerequisites

```bash
# System Requirements
- Python 3.9+
- Docker 20.10+
- Kubernetes 1.21+ (for production)
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for acceleration)

# Quantum Hardware (Optional)
- IBM Quantum Network access
- Xanadu photonic processors
- Diamond NV-center systems
```

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/danieleschmidt/qnet-no.git
cd qnet-no

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Run basic validation
python validate_algorithms.py

# 5. Start development server
python -m qnet_no.cli demo
```

### Docker Deployment

```bash
# Build image
docker build -t qnet-no:latest .

# Run container
docker run -p 8080:8080 \
  -e QNET_REGION=us-east-1 \
  -e QNET_LOCALE=en-US \
  qnet-no:latest

# Run with quantum hardware access
docker run -p 8080:8080 \
  -e IBM_QUANTUM_TOKEN=your_token \
  -e XANADU_API_KEY=your_key \
  qnet-no:latest
```

### Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: '3.8'
services:
  qnet-no:
    build: .
    ports:
      - "8080:8080"
    environment:
      - QNET_REGION=us-east-1
      - QNET_LOCALE=en-US
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - prometheus
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## ☸️ Production Kubernetes Deployment

### Namespace Setup

```bash
# Create namespace
kubectl create namespace qnet-no-prod

# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
```

### Core Deployment

```bash
# Deploy main application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Deploy monitoring
kubectl apply -f k8s/monitoring/

# Enable auto-scaling
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods -n qnet-no-prod
kubectl get svc -n qnet-no-prod
```

### Multi-Region Deployment

```bash
# Deploy to multiple regions
./scripts/deploy.sh production us-east-1 v1.0.0
./scripts/deploy.sh production eu-west-1 v1.0.0
./scripts/deploy.sh production ap-northeast-1 v1.0.0

# Verify deployments
kubectl get deployments --all-namespaces -l app=qnet-no
```

## 🌍 Global Configuration

### Region-Specific Settings

```python
# config/production.py
REGIONAL_CONFIGS = {
    'us-east-1': {
        'locale': 'en-US',
        'currency': 'USD', 
        'compliance': ['CCPA', 'SOC2'],
        'quantum_hardware': True,
        'entanglement_limit_km': 2000
    },
    'eu-west-1': {
        'locale': 'en-GB',
        'currency': 'EUR',
        'compliance': ['GDPR'],
        'quantum_hardware': True,
        'data_sovereignty': True,
        'entanglement_limit_km': 1500
    },
    'ap-northeast-1': {
        'locale': 'ja-JP',
        'currency': 'JPY',
        'compliance': ['PDPA'],
        'quantum_hardware': True,
        'entanglement_limit_km': 1200
    }
}
```

### Environment Variables

```bash
# Core Configuration
QNET_REGION=us-east-1
QNET_LOCALE=en-US
QNET_LOG_LEVEL=INFO
QNET_DEBUG=false

# Quantum Hardware
IBM_QUANTUM_TOKEN=your_token
XANADU_API_KEY=your_key
NV_CENTER_ENDPOINT=https://nv.example.com

# Database & Cache
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/qnet

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000
JAEGER_ENDPOINT=http://jaeger:14268

# Security
SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key
JWT_SECRET=your-jwt-secret

# Compliance
GDPR_ENABLED=true
CCPA_ENABLED=false
DATA_RETENTION_DAYS=365
```

## 🛡️ Security Configuration

### SSL/TLS Setup

```bash
# Generate certificates
./scripts/generate-certs.sh

# Apply TLS configuration
kubectl create secret tls qnet-no-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n qnet-no-prod

# Update ingress for HTTPS
kubectl apply -f k8s/ingress-tls.yaml
```

### RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: qnet-no-role
  namespace: qnet-no-prod
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "update"]
```

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qnet-no-network-policy
  namespace: qnet-no-prod
spec:
  podSelector:
    matchLabels:
      app: qnet-no
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'qnet-no'
    static_configs:
      - targets: ['qnet-no:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'quantum-metrics'
    static_configs:
      - targets: ['qnet-no:8080']
    metrics_path: '/quantum-metrics'
    scrape_interval: 10s
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "QNet-NO Performance",
    "panels": [
      {
        "title": "Quantum Advantage Score",
        "type": "stat",
        "targets": [
          {
            "expr": "qnet_quantum_advantage_score",
            "legendFormat": "QA Score"
          }
        ]
      },
      {
        "title": "Entanglement Quality",
        "type": "graph", 
        "targets": [
          {
            "expr": "qnet_entanglement_fidelity",
            "legendFormat": "Fidelity"
          }
        ]
      },
      {
        "title": "Task Scheduling Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(qnet_tasks_scheduled_total[5m])",
            "legendFormat": "Tasks/sec"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  quantum:
    format: '[%(asctime)s] %(levelname)s [%(name)s] [QA:%(quantum_advantage)s] %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: detailed
    level: INFO
  
  file:
    class: logging.FileHandler
    filename: qnet-no.log
    formatter: detailed
    level: DEBUG
  
  quantum_file:
    class: logging.FileHandler
    filename: quantum-metrics.log
    formatter: quantum
    level: INFO

root:
  level: INFO
  handlers: [console, file]

loggers:
  qnet_no.algorithms:
    level: DEBUG
    handlers: [quantum_file]
    propagate: false
```

## 🔧 Performance Tuning

### Resource Limits

```yaml
# k8s/deployment.yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
    nvidia.com/gpu: 1  # Optional GPU
```

### Auto-Scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qnet-no-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qnet-no
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: quantum_advantage_score
      target:
        type: AverageValue
        averageValue: "1.5"
```

### Quantum Hardware Optimization

```python
# config/quantum.py
QUANTUM_OPTIMIZATION = {
    'circuit_depth_limit': 100,
    'qubit_connectivity': 'heavy_hex',  # IBM topology
    'error_mitigation': True,
    'readout_error_correction': True,
    'gate_time_optimization': True,
    'decoherence_aware_scheduling': True,
    'entanglement_purification': True,
    'quantum_error_correction': False  # Enable for fault-tolerant QPUs
}

SCHEDULING_OPTIMIZATION = {
    'qaoa_layers': 4,
    'optimization_steps': 100,
    'classical_optimizer': 'COBYLA',
    'adaptive_parameters': True,
    'real_time_adaptation': True,
    'performance_monitoring_interval': 30  # seconds
}
```

## 🧪 Testing & Validation

### Automated Testing

```bash
# Run full test suite
python -m pytest tests/ -v --cov=qnet_no

# Run specific test categories
python -m pytest tests/test_hybrid_scheduling.py -v
python -m pytest tests/test_quantum_operators.py -v

# Run integration tests
python -m pytest tests/integration/ -v --slow

# Run performance benchmarks
python -m pytest tests/benchmarks/ -v --benchmark-only
```

### Load Testing

```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.42.0/k6-v0.42.0-linux-amd64.tar.gz -L | tar xvz

# Run load tests
k6 run tests/load/quantum-scheduling.js
k6 run tests/load/neural-operator-training.js

# Stress test quantum advantage
k6 run --duration=300s --rps=100 tests/load/quantum-advantage-stress.js
```

### Quantum Validation

```python
# Run quantum advantage validation
python research/validation_study.py

# Generate research results
python research/experimental_framework.py --experiment comprehensive --trials 100

# Validate against baselines  
python scripts/validate_quantum_advantage.py --statistical-significance 0.01
```

## 🚨 Troubleshooting

### Common Issues

1. **Quantum Hardware Connection Failed**
   ```bash
   # Check quantum credentials
   echo $IBM_QUANTUM_TOKEN | base64 -d
   
   # Test connection
   python -c "from qiskit import IBMQ; IBMQ.load_account(); print('Connected!')"
   ```

2. **Entanglement Quality Too Low**
   ```python
   # Increase fidelity threshold
   network.fidelity_threshold = 0.95
   
   # Enable error mitigation
   network.enable_error_mitigation = True
   ```

3. **Scheduling Performance Issues**
   ```python
   # Reduce QAOA layers for speed
   config.qaoa_layers = 2
   
   # Increase classical optimization steps
   config.optimization_steps = 200
   ```

4. **Memory Issues with Large Networks**
   ```python
   # Optimize Schmidt rank
   operator.schmidt_rank = min(16, optimal_rank)
   
   # Enable memory pooling
   operator.enable_memory_pooling = True
   ```

### Health Checks

```bash
# Application health
curl http://localhost:8080/health

# Quantum health
curl http://localhost:8080/quantum-health

# Detailed status
curl http://localhost:8080/status | jq '.'
```

### Logging & Debugging

```bash
# Follow logs
kubectl logs -f deployment/qnet-no -n qnet-no-prod

# Debug quantum operations
kubectl logs -f deployment/qnet-no -n qnet-no-prod | grep "quantum"

# Monitor performance
kubectl top pods -n qnet-no-prod
```

## 📈 Scaling Guidelines

### Horizontal Scaling

```yaml
# Scale replicas based on load
kubectl scale deployment qnet-no --replicas=5 -n qnet-no-prod

# Auto-scale based on quantum advantage
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-hpa
spec:
  metrics:
  - type: Object
    object:
      metric:
        name: quantum_advantage_score
      target:
        type: Value
        value: "2.0"
```

### Vertical Scaling

```yaml
# Increase resource limits
resources:
  limits:
    memory: "16Gi"
    cpu: "8000m"
    nvidia.com/gpu: 2
```

### Network Scaling

```python
# Scale quantum network
network.add_quantum_nodes(count=8)
network.optimize_topology()
network.distribute_entanglement()

# Scale classical resources
scheduler.enable_distributed_computing(additional_nodes=4)
```

## 🔒 Security Best Practices

1. **Quantum State Protection**
   - Encrypt quantum state representations
   - Secure entanglement key distribution
   - Implement quantum state privacy

2. **Classical-Quantum Interface Security**
   - Secure quantum measurement transmission
   - Authenticated quantum control channels
   - Encrypted parameter optimization

3. **Compliance & Auditing**
   - Enable GDPR quantum data protection
   - Implement quantum operation logging
   - Regular security audits

## 📞 Support & Maintenance

### Monitoring Alerts

```yaml
# monitoring/alerts.yml
groups:
  - name: qnet-no
    rules:
      - alert: QuantumAdvantageBelow1
        expr: qnet_quantum_advantage_score < 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Quantum advantage lost"
      
      - alert: EntanglementQualityLow
        expr: qnet_entanglement_fidelity < 0.8
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Entanglement quality degraded"
```

### Backup & Recovery

```bash
# Backup configuration
kubectl get configmaps -o yaml > backup/configmaps.yaml
kubectl get secrets -o yaml > backup/secrets.yaml

# Backup persistent data
kubectl exec -it postgres-pod -- pg_dump qnet > backup/database.sql

# Recovery
kubectl apply -f backup/
psql qnet < backup/database.sql
```

For additional support, please refer to:
- 📚 [API Documentation](https://docs.qnet-no.ai)
- 💬 [Community Discord](https://discord.gg/qnet-no)
- 🐛 [Issue Tracker](https://github.com/danieleschmidt/qnet-no/issues)
- 📧 [Enterprise Support](mailto:support@terragonlabs.ai)