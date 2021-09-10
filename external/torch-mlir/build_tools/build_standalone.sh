#!/bin/bash

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple script that does a CMake configure of this project as an external
# LLVM project so it can be tested in isolation to larger assemblies.
# This is meant for CI's and project maintainers.

set -eu -o errtrace

project_dir="$(cd $(dirname $0)/.. && pwd)"
llvm_project_dir="$project_dir/../llvm-project"
build_dir="$project_dir/build"

cmake -GNinja -B"$build_dir" "$llvm_project_dir/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$project_dir" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON

cd "$build_dir"
ninja tools/torch-mlir/all check-torch-mlir check-torch-mlir-plugin
