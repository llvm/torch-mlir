# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir
import torchvision.models as models

# TODO: Fix https://github.com/llvm/mlir-npcomp/issues/80
# XFAIL: *
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
print(mb.module)
