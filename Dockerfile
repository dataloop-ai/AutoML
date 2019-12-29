FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    wget \
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
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
    
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_AUTO_UPDATE_CONDA=false

# CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    pytorch=1.2 \
    torchvision=0.4.0 \
 && conda clean -ya

RUN conda install -c conda-forge pycocotools
# Install HDF5 Python bindings
RUN pip install \
	h5py==2.9.0 \
	h5py-cache==1.0 \
	Keras==2.1.6 \
	matplotlib==3.0.3 \
	numpy==1.17.1 \
	opencv-python==3.4.2.17 \
	pandas==0.24.2 \
	parse==1.12.1 \
	Pillow==6.2.0 \
	protobuf==3.9.1 \
	PyYAML==5.1.2 \
	requests==2.21.0 \
	scikit-image==0.15.0 \
	scipy==1.3.1 \
	tensorboard==2.0.1 \
	tensorflow==2.0.0 \
	tensorflow-gpu==2.0.0 \
	tqdm==4.32.2 \
	urllib3==1.24.3 \
	dtlpy==1.8.16 \
	https://storage.googleapis.com/dtlpy/dev/dtlpy_agent-1.9.1.0-py3-none-any.whl

WORKDIR /root

