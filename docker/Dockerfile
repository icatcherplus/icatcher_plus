FROM xychelsea/ffmpeg-nvidia:latest-jupyter

# Switch to root user
USER root

# Import missing GPG key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Update package lists
RUN apt-get update

# Install required packages
RUN apt-get install -y wget bzip2 ca-certificates

# Clean up the package cache
RUN apt-get clean

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
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
