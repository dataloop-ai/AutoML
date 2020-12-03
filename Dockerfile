FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN mkdir /root/data

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
    libgl1-mesa-glx \
    nano \
    vim \
    python3-numpy \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN cd /root/data && \
    git clone https://github.com/NoamRosenberg/tiny_coco.git

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
    cudatoolkit=10.2 \
    pytorch=1.6 \
    torchvision=0.7.0 \
 && conda clean -ya

RUN conda install -c conda-forge pycocotools
# Install HDF5 Python bindings

RUN cd /root && git clone https://github.com/dataloop-ai/AutoML.git \
    && mv /root/AutoML /root/ZazuML && cd /root/ZazuML

RUN pip install -r /root/ZazuML/requirements.txt

WORKDIR /root/ZazuML


# Add ssh in container 
# Set SSH(root) Password
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ARG PASSWORD=mikumiku
RUN echo root:${PASSWORD} | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
