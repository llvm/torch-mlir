# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.BatchNorm2d(123).to(dev)
result = model(torch.ones(42,123,4,5).to(dev))

# CHECK-LABEL: test_export_batchnorm
#       CHECK: aten.native_batch_norm
mlir = torch_mlir.get_mlir( result )
print("test_export_batchnorm")
print(mlir)
