#!/bin/bash
set -e

# Setup directories.
td="$(realpath $(dirname $0)/..)"
build_dir="$td/build"
install_mlir="$td/install-mlir"
build_mlir="$td/build-mlir"
declare -a extra_opts

if ! [ -d "$install_mlir/include/mlir" ]; then
  echo "MLIR install path does not appear valid: $install_mlir"
  exit 1
fi
mkdir -p "$build_dir"

# Make sure we are using python3.
python_exe="$(which python3)"
echo "Using python: $python_exe"
if [ -z "$python_exe" ]; then
  echo "Could not find python3"
  exit 1
fi

set -x
cmake -GNinja \
  "-H$td" \
  "-B$build_dir" \
  "-DCMAKE_BUILD_TYPE=Debug" \
  "-DCMAKE_CXX_FLAGS_DEBUG=-g3 -gdwarf-2 -Weverything -Werror" \
  "-DPYTHON_EXECUTABLE=$python_exe" \
  "-DMLIR_DIR=$install_mlir/lib/cmake/mlir" \
  "-DLLVM_EXTERNAL_LIT=$build_mlir/bin/llvm-lit" \
  "${extra_opts[@]}" \
  "$@"
