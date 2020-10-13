# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir
import torchvision.models as models

# TODO: Fix https://github.com/llvm/mlir-npcomp/issues/80
# XFAIL: *
# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

model = models.resnet18()
model.training = False

tensor = torch.randn(32,3,32,32)

mb = torch_mlir.ModuleBuilder()

with mb.capture_function("res18", [tensor]) as f:
  result = model(tensor)
  f.returns([result])

print(mb.module)

# for now we just check the output shape
# CHECK-LABEL: @res18
# TODO: Add checks once running to this point.
