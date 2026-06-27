# Production-grade Dockerfile for GraphCast Weather Forecasting Pipeline
# Targets NVIDIA CUDA 11.8.0 runtime with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Antigravity Weather Infrastructure <weather-infra@deepmind.google>"
LABEL description="Operational GraphCast model fine-tuning and inference container"

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for JAX and CUDA
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Install essential system dependencies (libgl1-mesa-glx for matplotlib/cartopy, eccodes for pygrib/grib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libeccodes-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    ca-certificates \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Put conda in path
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy the environment file into the container
WORKDIR /app
COPY environment.yml /app/environment.yml

# Create Conda environment
RUN conda env create -f /app/environment.yml && \
    conda clean -afy

# Set path to use the newly created environment
ENV PATH /opt/conda/envs/graphcast/bin:$PATH

# Copy project files
COPY . /app

# Expose ports for FastAPI (8000) and Flask Dashboard (5000)
EXPOSE 8000
EXPOSE 5000

# Default entrypoint starts the operational web dashboard and REST API
ENTRYPOINT ["python", "production_pipeline/app.py"]
