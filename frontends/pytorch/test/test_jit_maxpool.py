# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: python %s | FileCheck %s

model = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1),
                           dilation=1, return_indices=False, ceil_mode=False)

tensor = torch.randn(1,32,16,16)

# CHECK: PASS! fwd check
fwd_path = test.check_fwd(model, tensor)

