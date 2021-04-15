#!/bin/bash
# Usage (for in-tree build/ directory):
#   ./build_tools/install_mlir.sh
# For arbitrary build/install directories, set the env variables:
# - NPCOMP_BUILD_DIR
# - LLVM_BUILD_DIR
# - LLVM_INSTALL_DIR
set -e

portable_realpath() {
  # Create the directory if needed so that the `cd` doesn't fail.
  mkdir -p $1 && cd $1 && pwd
}

td="$(portable_realpath $(dirname $0)/..)"
build_dir="$(portable_realpath "${NPCOMP_BUILD_DIR:-$td/build}")"
build_mlir="${LLVM_BUILD_DIR-$build_dir/build-mlir}"
install_mlir="${LLVM_INSTALL_DIR-$build_dir/install-mlir}"

# Find LLVM source (assumes it is adjacent to this directory).
LLVM_SRC_DIR="$(portable_realpath "${LLVM_SRC_DIR:-$td/external/llvm-project}")"

if ! [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "Expected LLVM_SRC_DIR variable to be set correctly (got '$LLVM_SRC_DIR')"
  exit 1
fi
echo "Using LLVM source dir: $LLVM_SRC_DIR"
# Setup directories.
echo "Building MLIR in $build_mlir"
echo "Install MLIR to $install_mlir"
mkdir -p "$build_mlir"
mkdir -p "$install_mlir"

echo "Beginning build (commands will echo)"
set -x

function probe_python() {
  local python_exe="$1"
  local found
  local command
  command="import sys
if sys.version_info.major >= 3: print(sys.executable)"
  found="$("$python_exe" -c "$command")"
  if ! [ -z "$found" ]; then
    echo "$found"
  fi
}

python_exe=""
for python_candidate in python3 python; do
  python_exe="$(probe_python "$python_candidate")"
  if ! [ -z "$python_exe" ]; then
    break
  fi
done

echo "Using python: $python_exe"
if [ -z "$python_exe" ]; then
  echo "Could not find python3"
  exit 1
fi

cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_mlir" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
  "-DPython3_EXECUTABLE=$python_exe" \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DLLVM_INCLUDE_TOOLS=ON \
  "-DCMAKE_INSTALL_PREFIX=$install_mlir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DMLIR_BINDINGS_PYTHON_ENABLED=ON \
  "$@"

cmake --build "$build_mlir" --target install
