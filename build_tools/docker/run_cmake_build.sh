#!/usr/bin/env bash

torch_binary="${TORCH_BINARY:-ON}"


# Configure cmake to build torch-mlir in-tree
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$(pwd)" \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$(pwd)/externals/llvm-external-projects/torch-mlir-dialects" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DTORCH_MLIR_ENABLE_LTC=ON \
  -DTORCH_MLIR_USE_INSTALLED_PYTORCH="${torch_binary}" \
  -DPython3_EXECUTABLE="$(which python)" \
  externals/llvm-project/llvm

# Build torch-mlir
cmake --build build
