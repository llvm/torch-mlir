# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: python %s | FileCheck %s

model = torch.nn.LogSoftmax(dim=0)
tensor = torch.ones(1,2,3,4)

# CHECK: PASS! fwd check
fwd_path = test.check_fwd(model, tensor)
