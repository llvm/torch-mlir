#!/bin/bash
# Updates auto-generated shape library files for the `torch` dialect.
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
python_packages_dir="${build_dir}/tools/torch-mlir/python_packages"

TORCH_MLIR_EXT_PYTHONPATH="${TORCH_MLIR_EXT_PYTHONPATH:-""}"
pypath="${python_packages_dir}/torch_mlir"
if [ ! -z ${TORCH_MLIR_EXT_PYTHONPATH} ]; then
  pypath="${pypath}:${TORCH_MLIR_EXT_PYTHONPATH}"
fi
TORCH_MLIR_EXT_MODULES="${TORCH_MLIR_EXT_MODULES:-""}"
include_custom_op_example="${INCLUDE_CUSTOM_OP:-OFF}"
if [ "$include_custom_op_example" == "ON" ]; then
  ext_module="torch_mlir._torch_mlir_custom_op_example"
fi
if [ ! -z ${TORCH_MLIR_EXT_MODULES} ]; then
  ext_module="${ext_module},${TORCH_MLIR_EXT_MODULES} "
fi

PYTHONPATH="${pypath}" python \
  -m torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_lib_gen \
  --pytorch_op_extensions=${ext_module:-""} \
  --torch_transforms_cpp_dir="${torch_transforms_cpp_dir}" \
  --include_custom_op_example="${include_custom_op_example}"
