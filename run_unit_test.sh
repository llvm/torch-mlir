#!/usr/bin/env bash

# Run torch-mlir unit tests.
cmake --build build --target check-torch-mlir

# Run torch-mlir-python unit tests.
cmake --build build --target check-torch-mlir-python

# Run torch-mlir-dialects unit tests.
cmake --build build --target check-torch-mlir-dialects
