# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

ones = torch.ones(42,123,4,5)

with mb.capture_function("bn2d", [ones]) as f:
  model = torch.nn.BatchNorm2d(123)
  result = model(ones)
  f.returns([result])

# TODO: This test exercises promotion of const to arrays, inplace zero_ and
# add, all of which should be checked individually because they have specific
# behavior.
# CHECK-LABEL: @bn2d
# CHECK: %[[RESULT:.*]]:3 = torch.operator "aten.native_batch_norm"(%arg0
# CHECK: return %[[RESULT]]#0 : !torch.tensor<[42,123,4,5],f32>
print(mb.module)
