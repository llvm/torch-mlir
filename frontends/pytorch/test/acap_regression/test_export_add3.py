# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()
t0 = torch.randn((1,2,3,4), device=dev)
t1 = torch.randn((1,2,3,4), device=dev)
t2 = torch.randn((1,2,3,4), device=dev)

t3 = t0 + t1 + t2

#
# Generate and check the MLIR for the result tensor
#
t3_mlir = torch_mlir.get_mlir( t3 )

# CHECK-LABEL: test_export_add3
#   CHECK: %1 = "aten.add"(%arg0, %arg1, %0) {layer_name = "L0-add-0"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, i32) -> tensor<1x2x3x4xf32>
#   CHECK: %2 = "aten.add"(%1, %arg2, %0) {layer_name = "L1-add-1"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, i32) -> tensor<1x2x3x4xf32>
print("test_export_add3")
print(t3_mlir)
