FROM xychelsea/ffmpeg-nvidia:latest-jupyter

# Change user to root so can mkdir
USER root

# Set environment variables needed for Miniconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates && \
    apt-get clean && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Update the PATH environment variable
ENV PATH /opt/conda/bin:$PATH

# Install packages with conda
RUN conda install -y anaconda-client python=3.9 pip scikit-learn tensorboard pytest pingouin && \
    conda install -c conda-forge -y ffmpeg && \
    conda update -y conda && \
    conda clean --all --yes

# Install packages with pip
RUN pip install icatcher

# Create a directory for models and set environment variable
RUN mkdir /models
ENV ICATCHER_DATA_DIR=/models

# Download models so user doesn't have to
ADD docker/preload_models.py preload_models.py
RUN python preload_models.py
