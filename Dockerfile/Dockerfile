ARG BASE_IMG=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMG}

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y vim cmake g++ unzip git wget python3-pip && \
    pip install cmake --upgrade && \
    pip install pytorch

# Install Miniconda
RUN mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm -rf ~/miniconda3/miniconda.sh && \
apt-get install -y gnuplot && \
pip3 install numpy scipy

# Download and install Nsight Systems
# Install required dependencies
RUN apt-get update && \
apt-get install -y wget libxcb-cursor0 libnss3 libasound2 libopengl0 && \
rm -rf /var/lib/apt/lists/*
RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb && \
dpkg -i nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb && \
rm nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb

# Set environment variables for Miniconda and binaries
ENV PATH=/root/miniconda3/bin:$PATH
ENV PATH=/usr/local/bin:$PATH