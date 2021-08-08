#!/bin/bash
# Updates auto-generated ODS files for the `torch` dialect.
set -e

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${NPCOMP_BUILD_DIR:-$src_dir/build}")"
torch_ir_dir="${src_dir}/include/npcomp/Dialect/Torch/IR"

source $src_dir/.env
#ninja -C "${build_dir}"
python -m torch_mlir_utils.codegen.torch_ods_gen \
  --torch_ir_dir="${torch_ir_dir}" \
  --debug_registry_dump="${torch_ir_dir}/JITOperatorRegistryDump.txt"
