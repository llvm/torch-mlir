# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchvision.models as models
import torchvision.ops as tvops

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
    @annotate_args(
        [
            None,
            ([-1, 3, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([1, 3, 224, 224], torch.float32, True),
        ]
    )
    def forward(self, img):
        return self.resnet.forward(img)


@register_test_case(module_factory=lambda: ResNet18StaticModule())
def ResNet18StaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))


class IouOfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, 3, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, img):
        return self.mobilenetv3.forward(img)


@register_test_case(module_factory=lambda: MobilenetV3Module())
def MobilenetV3Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3, 224, 224))


# ==============================================================================
# torchvision.ops.nms tests
# ==============================================================================


class NmsModule(torch.nn.Module):
    """torchvision.ops.nms with well-formed boxes and partial suppression."""

    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4], torch.float32, True),
            ([6], torch.float32, True),
        ]
    )
    def forward(self, boxes, scores):
        return tvops.nms(boxes, scores, iou_threshold=0.5)


@register_test_case(module_factory=lambda: NmsModule())
def NmsModule_basic(module, tu: TestUtils):
    # 6 boxes with varying areas (0.5, 2.0, 3.0) so sort-rank != original index.
    # Score order: box3(0.95) > box0(0.90) > box1(0.75) > box2(0.60) > box4(0.50) > box5(0.30).
    # box3 suppresses box4; box0 suppresses box1 and box2; box5 isolated. Expected: [3, 0, 5].
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 2.0],  # score 0.90, area 2.0, overlaps with 1,2
            [0.1, 0.0, 1.1, 2.0],  # score 0.75, area 2.0, suppressed by 0
            [0.0, 0.1, 1.0, 2.1],  # score 0.60, area 2.0, suppressed by 0
            [10.0, 0.0, 11.0, 0.5],  # score 0.95, area 0.5, isolated from {0,1,2}
            [10.1, 0.0, 11.1, 0.5],  # score 0.50, area 0.5, suppressed by 3
            [50.0, 0.0, 51.0, 3.0],  # score 0.30, area 3.0, isolated
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.90, 0.75, 0.60, 0.95, 0.50, 0.30], dtype=torch.float32)
    module.forward(boxes, scores)
