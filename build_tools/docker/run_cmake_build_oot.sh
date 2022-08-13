#!/usr/bin/env bash

torch_binary="${TORCH_BINARY:-ON}"


# Configure cmake to build torch-mlir out-of-tree
cmake -GNinja -Bllvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python)" \
  externals/llvm-project/llvm

# Build llvm
cmake --build llvm-build

# TODO: Reenable LTC once OOT build is successful (https://github.com/llvm/torch-mlir/issues/1154)
cmake -GNinja -Bbuild \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LINKER=lld \
  -DLLVM_DIR="$(pwd)/llvm-build/lib/cmake/llvm/" \
  -DMLIR_DIR="$(pwd)/llvm-build/lib/cmake/mlir/" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DTORCH_MLIR_USE_INSTALLED_PYTORCH="${torch_binary}" \
  -DPython3_EXECUTABLE="$(which python)" \
  .

# Build torch-mlir
cmake --build build
