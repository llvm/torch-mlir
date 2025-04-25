#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Simple script that does a CMake configure of this project as an external
# LLVM project so it can be tested in isolation to larger assemblies.
# This is meant for CI's and project maintainers.

set -eu -o errtrace

project_dir="$(cd "$(dirname "$0")"/.. && pwd)"
llvm_project_dir="$project_dir/externals/llvm-project"
build_dir="$project_dir/build"

cmake -GNinja -B"$build_dir" "$llvm_project_dir/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$project_dir" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host

cd "$build_dir"
ninja tools/torch-mlir/all check-torch-mlir-all
