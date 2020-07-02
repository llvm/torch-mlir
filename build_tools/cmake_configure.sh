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
  DEBUG_FLAGS="-g3 -gdwarf2"
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
echo "Updating $td/.env file"
echo "PYTHONPATH=\"$(realpath "$build_dir/python_native"):$(realpath "$build_dir/python"):$(realpath "$build_dir/iree/bindings/python")\"" > "$td/.env"
echo "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1" >> "$td/.env"

set -x
cmake -GNinja \
  "-H$td" \
  "-B$build_dir" \
  "-DCMAKE_BUILD_TYPE=Debug" \
  "-DCMAKE_CXX_FLAGS_DEBUG=$DEBUG_FLAGS" \
  "-DPYTHON_EXECUTABLE=$python_exe" \
  "-DMLIR_DIR=$install_mlir/lib/cmake/mlir" \
  "-DLLVM_EXTERNAL_LIT=$LLVM_LIT" \
  "-DLLVM_ENABLE_WARNINGS=ON" \
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE" \
  "${extra_opts[@]}" \
  "$@"
