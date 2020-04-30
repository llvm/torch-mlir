#!/bin/bash
set -e
td="$(realpath $(dirname $0)/..)"

# Find LLVM source (assumes it is adjacent to this directory).
if [ -z "$LLVM_SRC_DIR" ]; then
  LLVM_SRC_DIR="$td/../llvm-project"
fi
LLVM_SRC_DIR="$(realpath "$LLVM_SRC_DIR")"

if ! [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "Expected LLVM_SRC_DIR variable to be set correctly (got '$LLVM_SRC_DIR')"
  exit 1
fi
echo "Using LLVM source dir: $LLVM_SRC_DIR"

# Setup directories.
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
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  "-DCMAKE_INSTALL_PREFIX=$install_mlir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \

cmake --build "$build_mlir" --target install
#cmake --build "$build_mlir" --target install
