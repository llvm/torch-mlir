#!/bin/bash
# Updates auto-generated ODS files for the `torch` dialect.
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
torch_ir_include_dir="${src_dir}/include/torch-mlir/Dialect/Torch/IR"

env_file_path="${src_dir}/.env"

if [[ ! -f "${env_file_path}" ]]; then
  echo "Couldn't find an env file at ${env_file_path}!"
  exit 1
fi

# Get PYTHONPATH from env file.
source $env_file_path
# Update PYTHONPATH with externals if specified.
TORCH_MLIR_EXT_PYTHONPATH="${TORCH_MLIR_EXT_PYTHONPATH:-""}"
if [ ! -z ${TORCH_MLIR_EXT_PYTHONPATH} ]; then
  PYTHONPATH="${PYTHONPATH}:${TORCH_MLIR_EXT_PYTHONPATH}"
fi
TORCH_MLIR_EXT_MODULES="${TORCH_MLIR_EXT_MODULES:-""}"
ext_module="${ext_module:-""}"
if [ ! -z ${TORCH_MLIR_EXT_MODULES} ]; then
  ext_module="${TORCH_MLIR_EXT_MODULES}"
fi

set +u
# To enable this python package, manually build torch_mlir with:
#   -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON
#   -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
# TODO: move this package out of JIT_IR_IMPORTER.
python3 \
  -m torch_mlir.jit_ir_importer.build_tools.torch_ods_gen \
  --torch_ir_include_dir="${torch_ir_include_dir}" \
  --pytorch_op_extensions="${ext_module}" \
  --debug_registry_dump="${torch_ir_include_dir}/JITOperatorRegistryDump.txt"
