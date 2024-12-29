
FROM python:3.9.19-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda create -n gawa python=3.9.0 && \
    conda install -n gawa -c conda-forge cfitsio=3.430 && \
    conda install -n gawa -c cta-observatory sparse2d && \
    conda install -n gawa jupyterlab ipykernel setuptools scikit-image=0.17.2 astropy=4.1 

WORKDIR /data

ENV CONDAPATH=/opt/conda/bin
ENV GAWA_ROOT=/data/
ENV PYTHONPATH=$PYTHONPATH:$GAWA_ROOT
ENV GAWA_LOG_LEVEL=info

COPY . .

RUN sed -i 's/dataclasses==0.8/dataclasses==0.6/g' requirements.txt && \ 
    conda run --no-capture-output -n gawa python -m pip install --upgrade debugpy && \
    conda run --no-capture-output -n gawa python -m pip install -r requirements.txt && \
    conda run --no-capture-output -n gawa python -m ipykernel install --user --name=gawa

RUN cp gawa.cfg.template gawa.cfg && \
    cp gawa.sh.template gawa.sh

EXPOSE 5678