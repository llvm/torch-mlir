#!/bin/bash
set -e
td="$(realpath $(dirname $0)/..)"

# Find LLVM source (assumes it is adjacent to this directory).
LLVM_SRC_DIR="$(realpath "${LLVM_SRC_DIR:-$td/external/llvm-project}")"

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

# TODO: Make it possible to build without an RTTI compiled LLVM. There are
# a handful of vague linkage issues that need to be fixed upstream.
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_mlir" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64;ARM" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  "-DCMAKE_INSTALL_PREFIX=$install_mlir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_RTTI=On

cmake --build "$build_mlir" --target install
