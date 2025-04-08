#!/usr/bin/env bash
set -eu -o errtrace

echo "Building torch-mlir"

cache_dir="${cache_dir:-}"
this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd ${this_dir}/../.. && pwd)"
build_dir="${build_dir:-}"


# Setup cache dir.
if [ -z "${cache_dir}" ]; then
  cache_dir="${repo_root}/.build-cache"
  mkdir -p "${cache_dir}"
  cache_dir="$(cd ${cache_dir} && pwd)"
fi

if [ -z "${build_dir}" ]; then
  build_dir="${repo_root}/build"
fi

echo "Building in ${build_dir}"

echo "Caching to ${cache_dir}"
mkdir -p "${cache_dir}/ccache"
mkdir -p "${cache_dir}/pip"

export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="350M"

# Clear ccache stats.
ccache -z

echo "::group::CMake configure"
cmake -GNinja -B"${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)" \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_LTC=OFF \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
  "${repo_root}"/externals/llvm-project/llvm
echo "::endgroup::"

echo "::group::Build"
cmake --build "${build_dir}"  --target tools/torch-mlir/all -- -k 0
echo "::endgroup::"

echo "Build completed successfully"

# echo "::group::Unit tests"
# cmake --build "${build_dir}" --target check-torch-mlir
# echo "::endgroup::"

# Show ccache stats.
ccache --show-stats
