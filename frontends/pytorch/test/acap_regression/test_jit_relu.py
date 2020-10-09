# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

model = torch.nn.ReLU()
tensor = torch.randn(10)

# CHECK: PASS! fwd check
fwd_path = test.check_ref(model, tensor)
