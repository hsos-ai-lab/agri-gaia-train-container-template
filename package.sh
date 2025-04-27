#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

PROVIDER="${1}"
CATEGORY="${2}"
ARCHITECTURE="${3}"

if [ -z "${PROVIDER}" ] || [ -z "${CATEGORY}" ] || [ -z "${ARCHITECTURE}" ]; then
  echo "Usage: ${0} <provider> <category> <architecture>" && exit 1
fi

ROOT_DIR="train-container-publish"

ARCHITECTURE_DIR="${ROOT_DIR}/${PROVIDER}/${CATEGORY}/${ARCHITECTURE}"
CONFIG_DIR="${ARCHITECTURE_DIR}/config"
DOCKER_DIR="${ARCHITECTURE_DIR}/docker"

rm -rf "${ROOT_DIR}" tmp "${ROOT_DIR}.zip"

cp -r dev tmp \
    && mkdir -p "${ARCHITECTURE_DIR}" "${CONFIG_DIR}" "${DOCKER_DIR}" \
    && cd tmp \
    && mv config.py config_to_args.py cvat_dataset.py dataset.py \
        docker-entrypoint.sh mount-s3.sh "../${ROOT_DIR}" \
    && mv config.jsonschema custom.json presets.json "../${CONFIG_DIR}" \
    && rm train_config.json export_config.json export.jsonschema export.py \
    && rm -rf minio train \
    && shopt -s dotglob \
    && mv ./* "../${DOCKER_DIR}"

cd .. && rm -rf tmp

[ -d "${ROOT_DIR}" ] \
  && zip -r "${ROOT_DIR}.zip" "${ROOT_DIR}" \
  && rm -rf "${ROOT_DIR}"
