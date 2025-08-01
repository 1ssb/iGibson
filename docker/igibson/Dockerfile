###############################################################################
# Base image with CUDA 11.3 & OpenGL
###############################################################################
FROM nvidia/cudagl:11.3.1-base-ubuntu20.04
ARG  DEBIAN_FRONTEND=noninteractive

###############################################################################
# 1. OS-level deps
###############################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
	cmake curl g++ git make vim wget libsm6 \
	cuda-command-line-tools-11-3 \
	&& rm -rf /var/lib/apt/lists/*

###############################################################################
# 2. Miniconda → /miniconda
###############################################################################
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	-o miniconda.sh && \
	bash miniconda.sh -b -p /miniconda && \
	rm miniconda.sh
ENV PATH=/miniconda/bin:$PATH

####################################################################
########################################
# 3.  Conda-forge-only env  (python 3.8)
########################################
RUN conda config --set ssl_verify false            && \
	conda config --remove channels defaults  || true && \
	conda config --add    channels conda-forge     && \
	conda config --set    channel_priority strict  && \
	conda create -y -n igibson python=3.8 pip      && \
	conda run   -n igibson pip install --upgrade pip

# Keep the env on PATH for all subsequent RUN/CMD
ENV PATH=/miniconda/envs/igibson/bin:$PATH

###############################################################################
# 4. PyTorch 1.10 (CUDA 11.3 wheels)  — MUST match base CUDA
###############################################################################
RUN conda run -n igibson pip install --no-cache-dir \
	torch==1.10.0+cu113 \
	torchvision==0.11.1+cu113 \
	torchaudio==0.10.0+cu113 \
	-f https://download.pytorch.org/whl/cu113/torch_stable.html

###############################################################################
# 5. Copy your existing repo into /src/iGibson and install it
###############################################################################
WORKDIR /src/iGibson
COPY . /src/iGibson           # build context = repo root

RUN pip install --no-cache-dir -e . && \
	pip install --no-cache-dir pytest "ray[default,rllib]" stable-baselines3 && \
	rm -rf /root/.cache

###############################################################################
# 6. Default working dir & shell
###############################################################################
WORKDIR /src/iGibson/igibson/examples
CMD ["/bin/bash"]



