#!/usr/bin/env bash

docker build -t torch-mlir-cmake:dev .

docker run -v $(pwd):/opt/src/torch-mlir/torch-mlir -it torch-mlir-cmake:dev
