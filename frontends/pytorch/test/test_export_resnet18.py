# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import torchvision.models as models

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = models.resnet18().to(dev)
model.training = False

tensor = torch.randn(32,3,32,32).to(dev)
result = model(tensor)

mlir = torch_mlir.get_mlir( result )

# for now we just check the output shape
# CHECK-LABEL: test_export_resnet18
#   CHECK: return %{{.*}} : tensor<32x1000xf32>
print("test_export_resnet18")
print(mlir)
