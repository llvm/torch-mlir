# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()

tensor = torch.randn(2,3).to(dev)
result = tensor.t()

ref_result = tensor.to('cpu').t()
# CHECK: PASS! transpose check
test.compare(ref_result, result, "transpose")
