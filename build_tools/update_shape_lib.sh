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

set -eo pipefail

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${TORCH_MLIR_BUILD_DIR:-$src_dir/build}")"
torch_transforms_cpp_dir="${src_dir}/lib/Dialect/Torch/Transforms"
python_packages_dir="${build_dir}/tools/torch-mlir/python_packages"

pypath="${python_packages_dir}/torch_mlir"
# TODO: Re-enable once custom op support is back.
#if [ ! -z ${TORCH_MLIR_EXT_PYTHONPATH} ]; then
#  pypath="${pypath}:${TORCH_MLIR_EXT_PYTHONPATH}"
#fi
#ext_module="torch_mlir._torch_mlir_custom_op_example"
#if [ ! -z ${TORCH_MLIR_EXT_MODULES} ]; then
#  ext_module="${ext_module},${TORCH_MLIR_EXT_MODULES} "
#fi

PYTHONPATH="${pypath}" python \
  -m torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_lib_gen \
  --torch_transforms_cpp_dir="${torch_transforms_cpp_dir}"

# TODO: Add back to shape_lib_gen invocation once custom op support is back.
#  --pytorch_op_extensions=${ext_module} \
