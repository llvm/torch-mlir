# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir
import torchvision.models as models

# XFAIL: *
# TODO: https://github.com/llvm/mlir-npcomp/issues/86
# TODO: Pass through npcomp-opt and FileCheck once able to elide large elements.
# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

model = models.resnet18()
model.training = False

tensor = torch.randn(32,3,32,32)

mb = torch_mlir.ModuleBuilder()

with mb.capture_function("res18", [tensor]) as f:
  result = model(tensor)
  f.returns([result])

# for now we just check the output shape
# CHECK-LABEL: @res18
# TODO: Add checks once running to this point.
# TODO: Enable printing once large elements can be elided (crashes lit).
# https://github.com/llvm/mlir-npcomp/issues/87
# print(mb.module)
