# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

# PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Tensorflow version 1 base image
# FROM nvcr.io/nvidia/tensorflow:22.08-tf1-py3

# Tensorflow version 2 base image
# FROM nvcr.io/nvidia/tensorflow:22.08-tf2-py3

# Disable questions during 'apt install'
ARG DEBIAN_FRONTEND=noninteractive

# Set Timezone to get correct output from `date`
ENV TZ=Europe/Berlin

# Copy the entrypoint script into the image
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Copy the s3fs mounting script into the container
COPY mount-s3.sh /mount-s3.sh
RUN chmod +x /mount-s3.sh

# Install essential packages inside the image
RUN apt update \
      && apt install -y --no-install-recommends \
      kmod fuse s3fs jq libxml2-utils curl wget tree \
      python3 python-is-python3 python3-pip

# Modify fuse.conf to allow users other than the owner to access mounted s3 bucket
RUN sed -i '/user_allow_other/s/^#//g' /etc/fuse.conf

# Update pip and install mandatory Python requirements
RUN pip install ujson==5.4.0 numpy==1.23.3 Shapely==1.8.4 \
      xmltodict==0.13.0 scikit-learn==1.1.2 pyyaml==6.0 \
      tf2onnx==1.14.0 onnxmltools==1.11.2

# Create an empty workspace directory
RUN mkdir -p /workspace
RUN rm -rf /workspace/*

# Change into workspace directory
WORKDIR /workspace

# Copy universal helper scripts into the image
COPY config_to_args.py config.py dataset.py cvat_dataset.py export.py ./

# Copy required train specific scripts into the image
COPY setup.py args.py train.py requirements.txt ./

# Copy configuration files into the image
COPY presets.json train_config.json export_config.json ./

# Install custom Python requirements
RUN pip install -r requirements.txt

# Copy .env into the image. This file will be sourced by the ENTRYPOINT.
COPY .env ./

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
CMD ["$@"]
