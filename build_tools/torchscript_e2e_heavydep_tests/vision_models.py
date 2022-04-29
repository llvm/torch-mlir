# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchvision.models as models

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
import timm

torch.manual_seed(0)


def getTracedRecursiveScriptModule(module):
    script_module = torch.jit.script(module)
    export(script_module.forward)
    annotate_args_decorator = annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    annotate_args_decorator(script_module.forward)
    return script_module


class VisionModule(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)


# ==============================================================================

resnet18_model = models.resnet18(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(resnet18_model)))
def Resnet18VisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

resnext50_32x4d_model = models.resnext50_32x4d(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(resnext50_32x4d_model)))
def Resnext50_32x4dVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

mnasnet1_0_model = models.mnasnet1_0(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(mnasnet1_0_model)))
def Mnasnet1_0VisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

alexnet_model = models.alexnet(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(alexnet_model)))
def AlexnetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

shufflenet_model = models.shufflenet_v2_x1_0(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(shufflenet_model)))
def ShufflenetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

squeezenet_model = models.squeezenet1_0(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(squeezenet_model)))
def SqueezenetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

vgg16_model = models.vgg16(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(vgg16_model)))
def Vgg16VisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

wide_resnet_model = models.wide_resnet50_2(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(wide_resnet_model)))
def Wide_ResnetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

efficientnet_model = models.efficientnet_b0(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(efficientnet_model)))
def EfficientnetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

mobilenet_v2_model = models.mobilenet_v2(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(mobilenet_v2_model)))
def Mobilenet_v2VisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

mobilenet_v3_large_model = models.mobilenet_v3_large(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(mobilenet_v3_large_model)))
def Mobilenet_v3_largeVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

resnet50_model = models.resnet50(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(resnet50_model)))
def Resnet50VisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

densenet121_model = models.densenet121(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(densenet121_model)))
def Densenet121VisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

timm_regnet_model = models.regnet_y_1_6gf(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(timm_regnet_model)))
def Timm_RegnetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

pytorch_unet_model = torch.hub.load(
    "mateuszbuda/brain-segmentation-pytorch",
    "unet",
    in_channels=3,
    out_channels=1,
    init_features=32,
    pretrained=True,
)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(pytorch_unet_model)))
def PytorchUnetVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

resnest_model = timm.create_model('resnest101e', pretrained=True)

input = torch.randn(1, 3, 224, 224)

@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(resnest_model)))
def ResnestVisionModel_basic(module, tu: TestUtils):
    module.forward(input)


# ==============================================================================

timm_vit_model = models.vit_b_16(pretrained=True)

input = torch.randn(1, 3, 224, 224)


@register_test_case(module_factory=lambda: getTracedRecursiveScriptModule(
    VisionModule(timm_vit_model)))
def ViTVisionModel_basic(module, tu: TestUtils):
    module.forward(input)
