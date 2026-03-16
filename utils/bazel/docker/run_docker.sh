#!/usr/bin/env bash

set -e

docker build -f utils/bazel/docker/Dockerfile \
             -t torch-mlir:dev \
             .

docker run -it \
           -v "$(pwd)":"/opt/src/torch-mlir" \
           -v "${HOME}/.cache/bazel":"/root/.cache/bazel" \
           torch-mlir:dev
