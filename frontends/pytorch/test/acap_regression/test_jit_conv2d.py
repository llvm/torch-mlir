# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: python %s | FileCheck %s

model = torch.nn.Conv2d(2,16,7,stride=[2,2], padding=[3,3],
                        dilation=1, groups=1, bias=True)

tensor = torch.randn((1,2,128,128))

# CHECK: PASS! fwd check
test.check_ref(model, tensor)
