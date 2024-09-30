#!/bin/bash

set -x
set -e

cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$(pwd)" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DTORCH_MLIR_USE_INSTALLED_PYTORCH=ON \
    -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
    -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
    -DTORCH_MLIR_ENABLE_LTC=OFF \
    $(pwd)/externals/llvm-project/llvm

cmake --build build --target TorchMLIRPythonModules TorchMLIRJITIRImporterPybind TorchMLIRE2ETestPythonModules check-torch-mlir-pt1 check-torch-mlir

