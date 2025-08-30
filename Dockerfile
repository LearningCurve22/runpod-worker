# CUDA 11.7 + Python 3.10, good for Torch 1.13.1 and building PyTorch3D.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (OpenCV, build tools, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip \
    git wget curl ca-certificates \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /workspace

# Python deps
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r requirements.txt \
 && python3 -m pip cache purge

# Clone upstream repo at build time for stability
RUN git clone https://github.com/ascust/3DMM-Fitting-Pytorch third_party/3DMM-Fitting-Pytorch

# App code
COPY app ./app
COPY handler.py ./handler.py

# Put repo on PYTHONPATH
ENV PYTHONPATH="/workspace/third_party/3DMM-Fitting-Pytorch:${PYTHONPATH}"

# Default command: start RunPod serverless handler
CMD ["python3", "-u", "/workspace/handler.py"]