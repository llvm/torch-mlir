# This build script is used by e2eshark test suite CI to build torch-mlir from source and the wheel
#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/.. && pwd)"
build_dir="$repo_root/build"
mkdir -p "$build_dir"
build_dir="$(cd $build_dir && pwd)"

python="$(which python)"
echo "Using python: $python"

export CMAKE_TOOLCHAIN_FILE="$this_dir/ci/linux_default_toolchain.cmake"

cd $repo_root

echo "::group::CMake configure"
cmake -S "$repo_root/externals/llvm-project/llvm" -B "$build_dir" \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host
echo "::endgroup::"

echo "::group::Build"
cmake --build "$build_dir"
echo "::endgroup::"

echo "::group::BuildWheel"
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel --dist-dir "$repo_root/torch-mlir-wheel" -v
echo "::endgroup::"
