# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch import nn

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

def get_quant_model_input():
    return 2 * torch.rand((1, 16)) - 1

def get_batched_quant_model_input():
    return 2 * torch.rand((1, 2, 16)) - 1

class QuantizedNoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @annotate_args([
        None,
        ([1, 16], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.dequantize(x)
        return x

def get_quantized_no_layer():
    model = QuantizedNoLayer()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(0)
    for _ in range(32):
        model(get_quant_model_input())
    torch.quantization.convert(model, inplace=True)
    return model

@register_test_case(module_factory=get_quantized_no_layer)
def QuantizedNoLayer_basic(module, tu: TestUtils):
    module.forward(get_quant_model_input())

class QuantizedSingleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.layers = nn.Sequential(
            nn.Linear(16, 8),
        )
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @annotate_args([
        None,
        ([1, 16], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.layers(x)
        x = self.dequantize(x)
        return x

def get_quantized_single_layer():
    model = QuantizedSingleLayer()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(0)
    for _ in range(32):
        model(get_quant_model_input())
    torch.quantization.convert(model, inplace=True)
    return model

@register_test_case(module_factory=get_quantized_single_layer)
def QuantizedSingleLayer_basic(module, tu: TestUtils):
    module.forward(get_quant_model_input())

class QuantizedBatchedInputSingleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.layers = nn.Sequential(
            nn.Linear(16, 8),
        )
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @annotate_args([
        None,
        ([1, 2, 16], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.layers(x)
        x = self.dequantize(x)
        return x

def get_batched_quantized_single_layer():
    model = QuantizedBatchedInputSingleLayer()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(0)
    for _ in range(32):
        model(get_batched_quant_model_input())
    torch.quantization.convert(model, inplace=True)
    return model

@register_test_case(module_factory=get_batched_quantized_single_layer)
def QuantizedBatchedInputSingleLayer_basic(module, tu: TestUtils):
    module.forward(get_batched_quant_model_input())

class QuantizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.layers = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @annotate_args([
        None,
        ([1, 16], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.layers(x)
        x = self.dequantize(x)
        return x

def get_quantized_mlp():
    model = QuantizedMLP()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(0)
    for _ in range(32):
        model(get_quant_model_input())
    torch.quantization.convert(model, inplace=True)
    return model

@register_test_case(module_factory=get_quantized_mlp)
def QuantizedMLP_basic(module, tu: TestUtils):
    module.forward(get_quant_model_input())

N = 1
Cin = 2
Cout = 3
Hin = 1
Win = 1
Hker = 1
Wker = 1

def get_conv_model_input():
    return torch.rand((N, Cin, Hin, Win))

class QuantizedConvTranspose2DModule(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(1)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=Cin,
                out_channels=Cout,
                kernel_size=(Hker, Wker),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                bias=True,
                output_padding=(0,0),
                dilation=1,
                ),
        )
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @annotate_args([
        None,
        ([N, Cin, Hin, Win], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.layers(x)
        x = self.dequantize(x)
        return x

def get_quantized_conv_transpose():
    model = QuantizedConvTranspose2DModule()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(1)
    for _ in range(32):
        model(get_conv_model_input())
    torch.quantization.convert(model, inplace=True)
    return model

@register_test_case(module_factory=get_quantized_conv_transpose)
def QuantizedConvTranspose2DModule_basic(module, tu: TestUtils):
    module.forward(0.5*torch.ones(N,Cin,Hin,Win,dtype=torch.float32))

class QuantizedConv2DModule(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(1)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=Cin,
                out_channels=Cout,
                kernel_size=(Hker, Wker),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                bias=True,
                dilation=1,
                ),
        )
        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    @export
    @annotate_args([
        None,
        ([N, Cin, Hin, Win], torch.float32, True),
    ])
    def forward(self, x):
        x = self.quantize(x)
        x = self.layers(x)
        x = self.dequantize(x)
        return x

def get_quantized_conv():
    model = QuantizedConv2DModule()
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    torch.manual_seed(1)
    for _ in range(32):
        model(get_conv_model_input())
    torch.quantization.convert(model, inplace=True)
    return model

@register_test_case(module_factory=get_quantized_conv)
def QuantizedConv2DModule_basic(module, tu: TestUtils):
    module.forward(get_conv_model_input())