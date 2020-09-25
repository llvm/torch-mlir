# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: python %s | FileCheck %s

model = torch.nn.LogSoftmax(dim=1)
tensor = torch.randn(3,5,requires_grad=True)

# CHECK: PASS! fwd check
fwd_path = test.check_fwd(model, tensor)

target = torch.tensor([1, 0, 4])
loss = torch.nn.NLLLoss()

# CHECK: PASS! back check
test.check_back(fwd_path, target, loss)
