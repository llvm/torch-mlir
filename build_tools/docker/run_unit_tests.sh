#!/usr/bin/env bash

# Full suite
# cmake --build build --target check-torch-mlir-all

# Run torch-mlir unit tests.
cmake --build build --target check-torch-mlir

# Run torch-mlir-python unit tests.
cmake --build build --target check-torch-mlir-python

# Run torch-mlir-dialects unit tests.
cmake --build build --target check-torch-mlir-dialects
