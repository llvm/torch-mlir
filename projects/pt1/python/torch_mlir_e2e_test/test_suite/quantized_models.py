# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch import nn
import torch.ao.quantization.fx._decomposed

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
    @annotate_args(
        [
            None,
            ([1, 16], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([1, 16], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([1, 2, 16], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([1, 16], torch.float32, True),
        ]
    )
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


# ==============================================================================


class FakeQuantizePerTensorAffineCachemaskModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([6, 4], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
            a, 2.0, 0, -128, 127
        )[0]


@register_test_case(module_factory=lambda: FakeQuantizePerTensorAffineCachemaskModule())
def FakeQuantizePerTensorAffineCachemaskModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6, 4))


# ==============================================================================


class QuantizePerTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 64, 112, 112], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale = 0.014940238557755947
        zp = -128
        quant_min = -128
        quant_max = 127
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, scale, zp, quant_min, quant_max, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizePerTensorModule())
def QuantizePerTensorModule_basic(module, tu: TestUtils):
    # use values within [-5, 5] to ensure we run into overflow/underflow
    module.forward(10 * torch.rand(1, 64, 112, 112) - 5)


# ==============================================================================


class QuantizedDecomposedDequantizePerTensor(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.int8, True),
        ]
    )
    def forward(self, x):
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.03, -10, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedDequantizePerTensor())
def QuantizedDecomposedDequantizePerTensor_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 8, low=-128, high=127).to(torch.int8))


# ==============================================================================


class QuantizedDecomposedQuantizePerTensor(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, 0.03, -10, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedQuantizePerTensor())
def QuantizedDecomposedQuantizePerTensor_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 8))


# ==============================================================================


class QuantizedDecomposedDequantizePerChannel(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.int8, True),
            ([8], torch.float32, True),
            ([8], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.dequantize_per_channel.default(
            x, scales, zero_points, 1, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedDequantizePerChannel())
def QuantizedDecomposedDequantizePerChannel_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 8, low=-128, high=127).to(torch.int8),
        tu.rand(8) + 0.01,
        tu.randint(8, low=-128, high=127).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedDequantizePerChannelUnsignedSymmetric(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.uint8, True),
            ([8], torch.float32, True),
        ]
    )
    def forward(self, x, scales):
        return torch.ops.quantized_decomposed.dequantize_per_channel.default(
            x, scales, None, 1, 0, 255, torch.uint8
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerChannelUnsignedSymmetric()
)
def QuantizedDecomposedDequantizePerChannelUnsignedSymmetric_basic(
    module, tu: TestUtils
):
    module.forward(
        tu.randint(4, 8, low=128, high=255).to(torch.uint8),
        tu.rand(8) + 0.01,
    )


# ==============================================================================


class QuantizedDecomposedQuantizePerChannel(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
            ([8], torch.float32, True),
            ([8], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.quantize_per_channel.default(
            x, scales, zero_points, 1, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedQuantizePerChannel())
def QuantizedDecomposedQuantizePerChannel_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(4, 8) - 5,
        tu.rand(8) + 0.01,
        tu.randint(8, low=-128, high=127).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedQuantizePerToken(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.float32, True),
            ([4, 1], torch.float32, True),
            ([4, 1], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.quantize_per_token.default(
            x, scales, zero_points, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedQuantizePerToken())
def QuantizedDecomposedQuantizePerToken_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(4, 16) - 5,
        tu.rand(4, 1) * 0.1 + 0.01,
        tu.randint(4, 1, low=-10, high=10).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedQuantizePerToken3D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 4, 16], torch.float32, True),
            ([2, 4, 1], torch.float32, True),
            ([2, 4, 1], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.quantize_per_token.default(
            x, scales, zero_points, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedQuantizePerToken3D())
def QuantizedDecomposedQuantizePerToken3D_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(2, 4, 16) - 5,
        tu.rand(2, 4, 1) * 0.1 + 0.01,
        tu.randint(2, 4, 1, low=-10, high=10).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedDequantizePerToken(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.int8, True),
            ([4, 1], torch.float32, True),
            ([4, 1], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.dequantize_per_token.default(
            x, scales, zero_points, -128, 127, torch.int8, torch.float32
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedDequantizePerToken())
def QuantizedDecomposedDequantizePerToken_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 16, low=-128, high=127).to(torch.int8),
        tu.rand(4, 1) * 0.1 + 0.01,
        tu.randint(4, 1, low=-10, high=10).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedDequantizePerToken3D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 4, 16], torch.int8, True),
            ([2, 4, 1], torch.float32, True),
            ([2, 4, 1], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.dequantize_per_token.default(
            x, scales, zero_points, -128, 127, torch.int8, torch.float32
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedDequantizePerToken3D())
def QuantizedDecomposedDequantizePerToken3D_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(2, 4, 16, low=-128, high=127).to(torch.int8),
        tu.rand(2, 4, 1) * 0.1 + 0.01,
        tu.randint(2, 4, 1, low=-10, high=10).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedChooseQparamsPerTokenAsymmetric(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                x, torch.int8
            )
        )
        return scale, zp


@register_test_case(
    module_factory=lambda: QuantizedDecomposedChooseQparamsPerTokenAsymmetric()
)
def QuantizedDecomposedChooseQparamsPerTokenAsymmetric_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(4, 16) - 5)


# ==============================================================================


class QuantizedDecomposedChooseQparamsPerTokenSymmetric(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams_per_token.default(
            x, torch.int8
        )
        return scale.to(torch.float32), zp.to(torch.float32)


@register_test_case(
    module_factory=lambda: QuantizedDecomposedChooseQparamsPerTokenSymmetric()
)
def QuantizedDecomposedChooseQparamsPerTokenSymmetric_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(4, 16) - 5)


# ==============================================================================


class QuantizedDecomposedDynamicQuantPerTokenAsymmetric(torch.nn.Module):
    """
    Mimics dynamic per-token asymmetric activation quantization as used in
    quantized LLM inference:
      1. choose_qparams_per_token_asymmetric: compute per-token scale/zp
      2. quantize_per_token: quantize activations to int8
      3. dequantize_per_token: dequantize back to float32
    """

    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                x, torch.int8
            )
        )
        xq = torch.ops.quantized_decomposed.quantize_per_token.default(
            x, scale, zp, -128, 127, torch.int8
        )
        return torch.ops.quantized_decomposed.dequantize_per_token.default(
            xq, scale, zp, -128, 127, torch.int8, torch.float32
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDynamicQuantPerTokenAsymmetric()
)
def QuantizedDecomposedDynamicQuantPerTokenAsymmetric_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(4, 16) - 5)


# ==============================================================================


class QuantizedDecomposedDynamicQuantPerTokenAsymmetric3D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 4, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                x, torch.int8
            )
        )
        xq = torch.ops.quantized_decomposed.quantize_per_token.default(
            x, scale, zp, -128, 127, torch.int8
        )
        return torch.ops.quantized_decomposed.dequantize_per_token.default(
            xq, scale, zp, -128, 127, torch.int8, torch.float32
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDynamicQuantPerTokenAsymmetric3D()
)
def QuantizedDecomposedDynamicQuantPerTokenAsymmetric3D_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(2, 4, 16) - 5)


# ==============================================================================


class QuantizedDecomposedDynamicQuantPerTokenSymmetric(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams_per_token.default(
            x, torch.int8
        )
        xq = torch.ops.quantized_decomposed.quantize_per_token.default(
            x, scale, zp, -128, 127, torch.int8
        )
        return torch.ops.quantized_decomposed.dequantize_per_token.default(
            xq, scale, zp, -128, 127, torch.int8, torch.float32
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDynamicQuantPerTokenSymmetric()
)
def QuantizedDecomposedDynamicQuantPerTokenSymmetric_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(4, 16) - 5)
