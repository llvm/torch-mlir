#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
torch_version="${1:-unknown}"

export CMAKE_TOOLCHAIN_FILE="$this_dir/linux_default_toolchain.cmake"
export CC=clang
export CXX=clang++
export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="350M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build $repo_root/build --target check-torch-mlir-all
