#!/bin/bash

# QNet-NO Production Deployment Script
set -e

echo "🚀 Starting QNet-NO production deployment..."

# Configuration
ENVIRONMENT=${1:-production}
NAMESPACE="qnet-no"
IMAGE_TAG=${2:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

# Build Docker image
build_image() {
    log_info "Building QNet-NO Docker image..."
    
    docker build -t qnet-no:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully ✓"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes cluster..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations in order
    log_info "Applying ConfigMaps and Secrets..."
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    
    log_info "Creating Persistent Volume Claims..."
    kubectl apply -f k8s/pvc.yaml
    
    log_info "Deploying applications..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    
    log_info "Setting up auto-scaling..."
    kubectl apply -f k8s/hpa.yaml
    
    log_info "Kubernetes deployment completed ✓"
}

# Deploy with Docker Compose (alternative)
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p logs cache data models
    
    # Start services
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log_info "Docker Compose deployment completed ✓"
    else
        log_error "Docker Compose deployment failed"
        exit 1
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        kubectl wait --for=condition=available --timeout=600s deployment/qnet-no-coordinator -n ${NAMESPACE}
        kubectl wait --for=condition=available --timeout=600s deployment/qnet-no-worker -n ${NAMESPACE}
    else
        # Wait for Docker Compose services
        timeout=300
        while [ $timeout -gt 0 ]; do
            if docker-compose ps | grep -q "Up"; then
                break
            fi
            sleep 5
            timeout=$((timeout - 5))
        done
        
        if [ $timeout -le 0 ]; then
            log_error "Deployment did not become ready in time"
            exit 1
        fi
    fi
    
    log_info "Deployment is ready ✓"
}

# Run health checks
health_check() {
    log_info "Running health checks..."
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        # Get service endpoint
        COORDINATOR_IP=$(kubectl get service qnet-no-external -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -z "$COORDINATOR_IP" ]; then
            COORDINATOR_IP=$(kubectl get service qnet-no-coordinator-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
        fi
        ENDPOINT="http://${COORDINATOR_IP}:8000"
    else
        ENDPOINT="http://localhost:8000"
    fi
    
    # Test health endpoint
    for i in {1..10}; do
        if curl -f "${ENDPOINT}/health" > /dev/null 2>&1; then
            log_info "Health check passed ✓"
            return 0
        fi
        log_warn "Health check attempt $i failed, retrying in 10 seconds..."
        sleep 10
    done
    
    log_error "Health checks failed"
    return 1
}

# Display deployment information
show_deployment_info() {
    log_info "Deployment Information:"
    echo "========================"
    echo "Environment: ${ENVIRONMENT}"
    echo "Namespace: ${NAMESPACE}"
    echo "Image Tag: ${IMAGE_TAG}"
    echo "Deployment Method: ${DEPLOYMENT_METHOD}"
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        echo ""
        echo "Kubernetes Resources:"
        kubectl get pods,services,hpa -n ${NAMESPACE}
        
        echo ""
        echo "External Access:"
        EXTERNAL_IP=$(kubectl get service qnet-no-external -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$EXTERNAL_IP" ]; then
            echo "API: http://${EXTERNAL_IP}:80"
            echo "Web Interface: http://${EXTERNAL_IP}:8080"
        else
            echo "Use 'kubectl port-forward' to access services locally"
        fi
    else
        echo ""
        echo "Docker Compose Services:"
        docker-compose ps
        
        echo ""
        echo "Local Access:"
        echo "API: http://localhost:8000"
        echo "Web Interface: http://localhost:8080"
        echo "Grafana: http://localhost:3000 (admin/quantum_viz_2024)"
        echo "Prometheus: http://localhost:9090"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        kubectl delete -f k8s/ --ignore-not-found=true
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    else
        docker-compose down -v
    fi
    
    log_info "Cleanup completed ✓"
}

# Main deployment logic
main() {
    # Determine deployment method
    if [ "$ENVIRONMENT" = "local" ]; then
        DEPLOYMENT_METHOD="docker-compose"
    else
        DEPLOYMENT_METHOD="kubernetes"
    fi
    
    log_info "Deployment method: ${DEPLOYMENT_METHOD}"
    
    # Handle cleanup flag
    if [ "$1" = "cleanup" ]; then
        cleanup
        exit 0
    fi
    
    # Run deployment steps
    check_prerequisites
    build_image
    
    if [ "$DEPLOYMENT_METHOD" = "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    wait_for_deployment
    
    if health_check; then
        show_deployment_info
        log_info "🎉 QNet-NO deployment completed successfully!"
    else
        log_error "Deployment completed but health checks failed"
        exit 1
    fi
}

# Handle script arguments
case "$1" in
    "cleanup")
        cleanup
        ;;
    "local"|"production"|"staging")
        main "$@"
        ;;
    *)
        echo "Usage: $0 {local|production|staging|cleanup} [image_tag]"
        echo ""
        echo "Examples:"
        echo "  $0 local                    # Deploy locally with Docker Compose"
        echo "  $0 production v1.0.0        # Deploy to production Kubernetes with tag v1.0.0"
        echo "  $0 cleanup                  # Clean up all deployments"
        exit 1
        ;;
esac