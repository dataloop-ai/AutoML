FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

WORKDIR /root
ADD . /root

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    nano \
    vim \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.8.3-Linux-x86_64.sh
    
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_AUTO_UPDATE_CONDA=false

# CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    pytorch=1.6 \
    torchvision=0.7.0 \
 && conda clean -ya

RUN conda install -c conda-forge pycocotools
# Install HDF5 Python bindings
RUN pip install -r requirements.txt