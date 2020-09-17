#!/bin/bash
# Runs lit-based tests by properly translating paths to the build directory.
# Example:
#   ./tools/run_lit.sh test/Dialect -v
set -e

td="$(realpath $(dirname $0)/..)"
build_dir="$td/build"
install_mlir="$td/install-mlir"
build_mlir="$td/external/llvm-project/build"

lit_exe="$build_mlir/bin/llvm-lit"
if ! [ -f "$lit_exe" ]; then
  echo "Could not find lit: $lit_exe"
  exit 1
fi

declare -a lit_args
for i in "$@"; do
  if [[ ${i:0:1} = "-" ]] || [[ ${i:0:1} = "/" ]]; then
    lit_args+=("$i")
  else
    if ! [ -e "$i" ]; then
      echo "Specified lit input does not exist: $i"
      exit 1
    fi
    test_local_path="$(realpath $i)"
    # Replace the src prefix with the build dir.
    test_build_path="$build_dir/${test_local_path##$td/}"
    lit_args+=("$test_build_path")
  fi
done

set -x
cd $build_dir
ninja npcomp-opt npcomp-run-mlir NPCOMPCompilerRuntimeShlib NPCOMPNativePyExt
cd test && python3 "$lit_exe" ${lit_args[@]}
