#!/bin/bash
set -e

# Find LLVM source.
if [ -z "$LLVM_SRC_DIR" ] || ! [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "Expected LLVM_SRC_DIR variable to be set correctly (got '$LLVM_SRC_DIR')"
  exit 1
fi
LLVM_SRC_DIR="$(realpath "$LLVM_SRC_DIR")"
echo "Using LLVM source dir: $LLVM_SRC_DIR"

# Setup directories.
td="$(realpath $(dirname $0)/..)"
build_mlir="$td/build-mlir"
install_mlir="$td/install-mlir"
echo "Building MLIR in $build_mlir"
echo "Install MLIR to $install_mlir"
mkdir -p "$build_mlir"
mkdir -p "$install_mlir"

echo "Beginning build (commands will echo)"
set -x

cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_mlir" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  "-DCMAKE_INSTALL_PREFIX=$install_mlir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \

cmake --build "$build_mlir"
cmake --build "$build_mlir" --target install
