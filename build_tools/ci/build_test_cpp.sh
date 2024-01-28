#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
build_dir="$repo_root/build"
install_dir="$repo_root/install"
mkdir -p "$build_dir"
build_dir="$(cd $build_dir && pwd)"
cache_dir="${cache_dir:-}"

# Setup cache dir.
if [ -z "${cache_dir}" ]; then
  cache_dir="${repo_root}/.build-cache"
  mkdir -p "${cache_dir}"
  cache_dir="$(cd ${cache_dir} && pwd)"
fi
echo "Caching to ${cache_dir}"
mkdir -p "${cache_dir}/ccache"
mkdir -p "${cache_dir}/pip"

python="$(which python)"
echo "Using python: $python"

export CMAKE_TOOLCHAIN_FILE="$this_dir/linux_default_toolchain.cmake"
export CC=clang
export CXX=clang++
export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="700M"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clear ccache stats.
ccache -z

echo "Configuring torch-mlir"
echo "======================"

cd $repo_root
# -DTORCH_MLIR_USE_INSTALLED_PYTORCH="$torch_from_bin" \
# -DTORCH_MLIR_SRC_PYTORCH_REPO=${TORCH_MLIR_SRC_PYTORCH_REPO} \
# -DTORCH_MLIR_SRC_PYTORCH_BRANCH=${TORCH_MLIR_SRC_PYTORCH_BRANCH} \
# -DTM_PYTORCH_INSTALL_WITHOUT_REBUILD=${TM_PYTORCH_INSTALL_WITHOUT_REBUILD} \

cmake -S "$repo_root/externals/llvm-project/llvm" -B "$build_dir" \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(which python)" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$repo_root" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DTORCH_MLIR_ENABLE_LTC=ON


echo "Building all"
echo "------------"
cmake --build "$build_dir" --target tools/torch-mlir/all -- -k 0

# echo "Installing"
# echo "----------"
# echo "Install to: $install_dir"
# cmake --build "$build_dir" --target iree-install-dist

# Show ccache stats.
ccache --show-stats
