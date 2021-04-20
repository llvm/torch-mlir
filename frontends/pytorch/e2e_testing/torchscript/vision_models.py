#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torchvision.models as models

from torch_mlir.torchscript.e2e_test.framework import TestUtils
from torch_mlir.torchscript.e2e_test.registry import register_test_case
from torch_mlir.torchscript.annotations import annotate_args, export

# ==============================================================================

class Resnet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.resnet = models.resnet18()
    @export
    @annotate_args([
        None,
        ([-1, 3, -1, -1], torch.float32),
    ])
    def forward(self, img):
        return self.resnet.forward(img)

@register_test_case(module_factory=lambda: Resnet18Module())
def Resnet18Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))
