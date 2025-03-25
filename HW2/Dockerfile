# Base image with CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential tools
RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        vim \
        build-essential \
        software-properties-common \
        tmux \
        htop \
        unzip \
        libgl1-mesa-glx \
        bzip2 && \
    apt-get clean

# Set a default shell for the container
SHELL ["/bin/bash", "-c"]

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Set environment variables for Conda
ENV PATH="/opt/miniconda/bin:$PATH"

# Verify Conda installation
RUN conda --version

# Entry point
CMD ["/bin/bash"]                                                                                                                                                                              
