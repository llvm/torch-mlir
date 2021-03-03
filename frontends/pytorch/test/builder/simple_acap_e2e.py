# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# RUN: %PYTHON %s | npcomp-opt -aten-recognize-kernels -numpy-public-functions-to-tensor -canonicalize | FileCheck %s
# TODO: Re-enable after adding support for 4-operand aten::add in `aten-recognize-kernels`.
# XFAIL: *

# TODO: This test should go away or become part of an e2e test suite. It is
# preserved right now as a stop-gap.

import torch
import torch_mlir

t0 = torch.randn((1,4))
t1 = torch.randn((4,1))

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("foobar", [t0, t1]) as f:
  result = t0 + t1
  f.returns([result])

# CHECK-LABEL:   func @foobar(
# CHECK-SAME:                 %[[VAL_0:.*]]: tensor<1x4xf32>,
# CHECK-SAME:                 %[[VAL_1:.*]]: tensor<4x1xf32>) -> tensor<4x4xf32> {
# CHECK:           %[[VAL_2:.*]] = constant 1 : i64
# CHECK:           %[[VAL_3:.*]] = "aten.add"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (tensor<1x4xf32>, tensor<4x1xf32>, i64) -> tensor<4x4xf32>
# CHECK:           return %[[VAL_3]] : tensor<4x4xf32>
# CHECK:         }
mb.module.operation.print(large_elements_limit=2)
