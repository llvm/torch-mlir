#!/bin/bash
# Updates the ATen dialect generated code from the PyTorch op registry.
# Requires that the project has been built and that PyTorch support is enabled.
set -e

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${NPCOMP_BUILD_DIR:-$src_dir/build}")"
aten_dir="${src_dir}/include/npcomp/Dialect/ATen/IR"

export PYTHONPATH="${build_dir}/python"

python -m torch_mlir_utils.codegen.torch_signature_ods_gen \
  --ods_td_file="${aten_dir}/GeneratedATenOps.td" \
  --ods_impl_file="${aten_dir}/GeneratedATenOps.cpp.inc" \
  --debug_op_reg_file="${aten_dir}/ATenOpRegistrations.txt"
