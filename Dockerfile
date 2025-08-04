# Production Dockerfile for QNet-NO Quantum Neural Operators
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QNET_ENV=production
ENV QNET_LOG_LEVEL=INFO
ENV QNET_LOG_DIR=/app/logs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install JAX with CUDA support
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/models

# Set up logging configuration
ENV PYTHONPATH=/app

# Install the package in development mode
RUN pip3 install -e .

# Create non-root user for security
RUN useradd -m -u 1000 qnet && \
    chown -R qnet:qnet /app
USER qnet

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import qnet_no; print('Health check passed')" || exit 1

# Expose ports for distributed computing
EXPOSE 8000 8001 8002 8003

# Default command
CMD ["python3", "-m", "qnet_no.examples.distributed_training"]