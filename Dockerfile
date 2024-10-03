# Use NVIDIA CUDA base image with Ubuntu 22.04, which includes CUDA 12.2 and cuDNN
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment to non-interactive to avoid timezone and other configuration prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies including OpenCV, wget, curl, Python3 development headers, and X11 libraries
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    python3-dev \
    python3-pip \
    gnupg2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    x11-apps \
    && apt-get clean

# Upgrade pip and install Python dependencies, including TensorRT via pip
RUN pip install --upgrade pip && \
    pip install onnxruntime-gpu nvidia-pyindex nvidia-tensorrt

# Install PyTorch with CUDA support (for CUDA 12.2)
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122

# Install OpenCV with GUI support for `imshow`
RUN pip install opencv-python

# Install any other required Python dependencies from a requirements.txt file
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Set the working directory
WORKDIR /code

# Copy code
COPY . /code/

# Set environment variables for NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video
