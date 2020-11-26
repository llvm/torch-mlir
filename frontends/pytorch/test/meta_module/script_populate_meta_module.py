# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

mb = torch_mlir.ModuleBuilder()

@mb.import_function
@torch.jit.script
def ndarray_arg_result(t0, t1, t2):
  return t0 + t1 + t2

# CHECK-LABEL: @ndarray_arg_result
# CHECK: >> Symbol Table:
# CHECK: 'ndarray_arg_result' -> generic func @ndarray_arg_result signature (NdArray, NdArray, NdArray) -> NdArray:
print(f"@ndarray_arg_result:\n{mb.meta_module}")
