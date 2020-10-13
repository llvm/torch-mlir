# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# See bug references below and remove XFAIL when resolved.
# XFAIL: *
# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# TODO: Both of these fail with the "unsupported from an unboxed API yet" error.
# The corresponding ops need to be manually coded. Then these can be moved into
# the capture. https://github.com/llvm/mlir-npcomp/issues/78
# TODO: These also create constant tensors (needs implementation of import of
# DenseElements constants). https://github.com/llvm/mlir-npcomp/issues/79
model = torch.nn.BatchNorm2d(123)
ones = torch.ones(42,123,4,5)

with mb.capture_function("bn2d", []) as f:
  result = model(ones)
  f.returns([result])

# CHECK-LABEL: @bn2d
print(mb.module)
