#!/usr/bin/env bash
set -eo pipefail

echo "Building torch-mlir"

cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$PWD/externals/llvm-external-projects/torch-mlir-dialects" \
  -DPython3_EXECUTABLE="$(which python)" \
  $GITHUB_WORKSPACE/externals/llvm-project/llvm

cmake --build build --config Release

echo "Build completed successfully"
