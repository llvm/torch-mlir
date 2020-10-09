# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.Linear(1024,16).to(dev)
tensor = torch.randn(4,1024).to(dev)

# CHECK: PASS! fwd check
fwd_path = test.check_fwd(model, tensor)
