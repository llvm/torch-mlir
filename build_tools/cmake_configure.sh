#!/bin/bash
# Configures the project with default options.
# LLVM/MLIR should be installed into the build directory first by running
# ./build_tools/install_mlir.sh.
#
# Usage (for in-tree build/ directory):
#   ./build_tools/cmake_configure.sh [ARGS...]
# Usage (for arbitrary build/ directory):
#   BUILD_DIR=/build ./build_tools/cmake_configure.sh [ARGS...]
set -e

portable_realpath() {
  echo "$(cd $1 && pwd)"
}

# Setup directories.
td="$(portable_realpath $(dirname $0)/..)"
build_dir="$(portable_realpath "${NPCOMP_BUILD_DIR:-$td/build}")"
build_mlir="${LLVM_BUILD_DIR-$build_dir/build-mlir}"
install_mlir="${LLVM_INSTALL_DIR-$build_dir/install-mlir}"
declare -a extra_opts

if ! [ -d "$install_mlir/include/mlir" ]; then
  echo "MLIR install path does not appear valid: $install_mlir"
  exit 1
fi
mkdir -p "$build_dir"

# Make sure we are using python3.
function probe_python() {
  local python_exe="$1"
  local found
  local command
  command="import sys
if sys.version_info.major >= 3: print(sys.executable)"
  set +e
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

# Detect windows.
if (which cygpath 2>/dev/null); then
  echo "Using windows path mangling and flags"
  DEBUG_FLAGS=""
  function translate_path() {
    cygpath --windows "$1"
  }
else
  DEBUG_FLAGS="-g3 -gdwarf-2"
  function translate_path() {
    echo "$1"
  }
fi

# Find llvm-lit.
LLVM_LIT=""
for candidate_lit in "$build_mlir/bin/llvm-lit" "$build_mlir/bin/llvm-lit.py"
do
  if [ -f "$candidate_lit" ]; then
    LLVM_LIT="$candidate_lit"
    break
  fi
done

if [ -z "$LLVM_LIT" ]; then
  echo "WARNING: Unable to find llvm-lit"
fi
echo "Using llvm-lit: $LLVM_LIT"

# Write a .env file for python tooling.
function write_env_file() {
  echo "Updating $build_dir/.env file"
  echo "PYTHONPATH=\"$(portable_realpath "$build_dir/python"):$(portable_realpath "$install_mlir/python")\"" > "$build_dir/.env"
  echo "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1" >> "$build_dir/.env"
  if ! cp "$build_dir/.env" "$td/.env"; then
    echo "WARNING: Failed to write $td/.env"
  fi
}
write_env_file

set -x
cmake -GNinja \
  "-H$td" \
  "-B$build_dir" \
  "-DCMAKE_BUILD_TYPE=Debug" \
  "-DNPCOMP_USE_SPLIT_DWARF=ON" \
  "-DCMAKE_CXX_FLAGS_DEBUG=$DEBUG_FLAGS" \
  "-DPYTHON_EXECUTABLE=$python_exe" \
  "-DMLIR_DIR=$install_mlir/lib/cmake/mlir" \
  "-DLLVM_EXTERNAL_LIT=$LLVM_LIT" \
  "-DLLVM_ENABLE_WARNINGS=ON" \
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE" \
  "${extra_opts[@]}" \
  "$@"
