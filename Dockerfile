# syntax = docker/dockerfile:experimental
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive
# Makes Poetry behave more like npm, with deps installed inside a .venv folder
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY ./install_cuda.sh ./install_cuda.sh
RUN ./install_cuda.sh && \
    rm install_cuda.sh

# System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip

# This is the "archive" branch of yolov3 (before the yolov5 changes were merged in)
RUN git clone https://github.com/edgeimpulse/yolov3 && \
    cd yolov3 && \
    git checkout 98068ef
RUN --mount=type=cache,target=/root/.cache/pip \
    cd yolov3 && pip3 install -r requirements.txt

# Install TensorFlow
COPY install_tensorflow.sh install_tensorflow.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Grab yolov3-tiny pretrained weights
RUN wget -O yolov3-tiny.weights https://cdn.edgeimpulse.com/build-system/yolov3-tiny.weights

# Local dependencies
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Export ONNX
RUN sed -i -e "s/ONNX_EXPORT = False/ONNX_EXPORT = True/" /app/yolov3/models.py

# Convert the weights into PyTorch weights
RUN cd yolov3 && \
    python3  -c "from models import *; convert('cfg/yolov3-tiny.cfg', '../yolov3-tiny.weights')"

# Download some files that are pulled in, so we can run w/o network access
RUN mkdir -p /root/.config/Ultralytics/ && wget -O /root/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf

# Remove the .git directory, otherwise it tries to fetch something (and we don't have network access)
RUN rm -rf /app/yolov3/.git

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
