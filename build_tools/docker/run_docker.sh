#!/usr/bin/env bash

docker build -f build_tools/docker/Dockerfile \
             -t torch-mlir-cmake:dev \
             .

docker run -it \
           -v "$(pwd)":/opt/src/torch-mlir \
           -e CCACHE_DIR=/opt/src/torch-mlir/.ccache \
           torch-mlir-cmake:dev
