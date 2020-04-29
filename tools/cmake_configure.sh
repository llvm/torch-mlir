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

# Detect linker.
# TODO: Generalize this.
for probe_linker in /usr/bin/ld.lld-10; do
  if which ld.lld-10; then
    echo "Using linker $probe_linker"
    extra_opts+=("-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=$probe_linker")
    break
  fi
done

set -x
cmake -GNinja \
  "-H$td" \
  "-B$build_dir" \
  "-DPYTHON_EXECUTABLE=$python_exe" \
  "-DMLIR_DIR=$install_mlir/lib/cmake/mlir" \
  "-DLLVM_EXTERNAL_LIT=$build_mlir/bin/llvm-lit" \
  "${extra_opts[@]}" \
  "$@"
