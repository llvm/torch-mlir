# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchvision.models as models

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.resnet = models.resnet18()
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, 3, -1, -1], torch.float32, True),
    ])
    def forward(self, img):
        return self.resnet.forward(img)


@register_test_case(module_factory=lambda: ResNet18Module())
def ResNet18Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))


class ResNet18StaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.resnet = models.resnet18()
        self.train(False)

    @export
    @annotate_args([
        None,
        ([1, 3, 224, 224], torch.float32, True),
    ])
    def forward(self, img):
        return self.resnet.forward(img)


@register_test_case(module_factory=lambda: ResNet18StaticModule())
def ResNet18StaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))


class IouOfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, bbox1, bbox2):
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
        lt = torch.maximum(bbox1[:, :2], bbox2[:, :2])
        rb = torch.minimum(bbox1[:, 2:], bbox2[:, 2:])

        overlap_coord = (rb - lt).clip(0)
        overlap = overlap_coord[:, 0] * overlap_coord[:, 1]
        union = area1 + area2 - overlap

        return overlap / union


@register_test_case(module_factory=lambda: IouOfModule())
def IouOfModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 4), tu.rand(1024, 4))


class MobilenetV3Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.mobilenetv3 = models.mobilenet_v3_small()
        self.train(False)

    @export
    @annotate_args([
        None,
        ([-1, 3, -1, -1], torch.float32, True),
    ])
    def forward(self, img):
        return self.mobilenetv3.forward(img)


@register_test_case(module_factory=lambda: MobilenetV3Module())
def MobilenetV3Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))
