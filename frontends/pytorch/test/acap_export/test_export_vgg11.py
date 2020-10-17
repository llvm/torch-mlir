# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir
import torchvision.models as models

# XFAIL: *
# TODO: https://github.com/llvm/mlir-npcomp/issues/86
# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

model = models.vgg11_bn()
model.training = False

inputs = torch.ones(32,3,32,32)

mb = torch_mlir.ModuleBuilder()

with mb.capture_function("vgg11", [inputs]) as f:
  result = model(inputs)
  f.returns([result])

# CHECK-LABEL: func @vgg11
# TODO: Add checks once passing this far.
# TODO: Enable printing once large elements can be elided (crashes lit).
# https://github.com/llvm/mlir-npcomp/issues/87
# print(mb.module)
