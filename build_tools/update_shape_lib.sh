#!/bin/bash
# Updates auto-generated shape library files for the `torch` dialect.
set -e

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${TORCH_MLIR_BUILD_DIR:-$src_dir/build}")"
torch_transforms_cpp_dir="${src_dir}/lib/Dialect/Torch/Transforms"
python_packages_dir="${build_dir}/tools/torch-mlir/python_packages"

#ninja -C "${build_dir}"
PYTHONPATH="${python_packages_dir}/torch_mlir" python \
  -m torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_lib_gen \
  --torch_transforms_cpp_dir="${torch_transforms_cpp_dir}"
