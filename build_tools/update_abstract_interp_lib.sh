#!/bin/bash
# Updates auto-generated abstract interpretation library files for the
# `torch` dialect.
#
# Environment variables:
#   TORCH_MLIR_EXT_MODULES: comma-separated list of python module names
#     which register custom PyTorch operators upon being imported.
#   TORCH_MLIR_EXT_PYTHONPATH: colon-separated list of paths necessary
#     for importing PyTorch extensions specified in TORCH_MLIR_EXT_MODULES.
# For more information on supporting custom operators, see:
#   ${TORCH_MLIR}/python/torch_mlir/_torch_mlir_custom_op_example/README.md

set -euo pipefail

src_dir="$(realpath "$(dirname "$0")"/..)"
build_dir="$(realpath "${TORCH_MLIR_BUILD_DIR:-$src_dir/build}")"
torch_transforms_cpp_dir="${src_dir}/lib/Dialect/Torch/Transforms"

in_tree_pkg_dir="${build_dir}/tools/torch-mlir/python_packages"
out_of_tree_pkg_dir="${build_dir}/python_packages"

if [[ ! -d "${in_tree_pkg_dir}" && ! -d "${out_of_tree_pkg_dir}" ]]; then
  echo "Couldn't find in-tree or out-of-tree build, exiting."
  exit 1
fi

# The `-nt` check works even if one of the two directories is missing.
if [[ "${in_tree_pkg_dir}" -nt "${out_of_tree_pkg_dir}" ]]; then
  python_packages_dir="${in_tree_pkg_dir}"
else
  python_packages_dir="${out_of_tree_pkg_dir}"
fi

TORCH_MLIR_EXT_PYTHONPATH="${TORCH_MLIR_EXT_PYTHONPATH:-""}"
pypath="${python_packages_dir}/torch_mlir"
if [ ! -z ${TORCH_MLIR_EXT_PYTHONPATH} ]; then
  pypath="${pypath}:${TORCH_MLIR_EXT_PYTHONPATH}"
fi
TORCH_MLIR_EXT_MODULES="${TORCH_MLIR_EXT_MODULES:-""}"
if [ ! -z ${TORCH_MLIR_EXT_MODULES} ]; then
  ext_module="${TORCH_MLIR_EXT_MODULES} "
fi

# To enable this python package, manually build torch_mlir with:
#   -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
# TODO: move this package out of JIT_IR_IMPORTER.
PYTHONPATH="${pypath}" python \
  -m torch_mlir.jit_ir_importer.build_tools.abstract_interp_lib_gen \
  --pytorch_op_extensions=${ext_module:-""} \
  --torch_transforms_cpp_dir="${torch_transforms_cpp_dir}"
