# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import torchvision.models as models

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = models.vgg11_bn().to(dev)
model.training = False

result = model(torch.ones(32,3,32,32).to(dev))

mlir = torch_mlir.get_mlir( result )

# for now we just check the output shape
# CHECK-LABEL: test_export_vgg11
#   CHECK: return %{{.*}} : tensor<32x1000xf32>
print("test_export_vgg11")
print(mlir)
