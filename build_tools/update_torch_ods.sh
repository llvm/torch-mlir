#!/bin/bash
# Updates auto-generated ODS files for the `torch` dialect.
set -e

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${TORCH_MLIR_BUILD_DIR:-$src_dir/build}")"
torch_ir_dir="${src_dir}/include/torch-mlir/Dialect/Torch/IR"
python_packages_dir="${build_dir}/tools/torch-mlir/python_packages"

#ninja -C "${build_dir}"
PYTHONPATH="${python_packages_dir}/torch_mlir" python \
  -m torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen \
  --torch_ir_dir="${torch_ir_dir}" \
  --debug_registry_dump="${torch_ir_dir}/JITOperatorRegistryDump.txt"
