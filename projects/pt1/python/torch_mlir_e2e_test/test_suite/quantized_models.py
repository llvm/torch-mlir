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


class QuantizedDecomposedDequantizePerTensorUnsigned(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.uint8, True),
        ]
    )
    def forward(self, x):
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.03, 200, 0, 255, torch.uint8
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerTensorUnsigned()
)
def QuantizedDecomposedDequantizePerTensorUnsigned_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 8, low=0, high=256).to(torch.uint8))


# ==============================================================================


class QuantizedDecomposedQuantizePerTensorUnsigned(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        # The refbackend returns every i8 buffer as int8, so widen the uint8
        # result to compare its values rather than its reinterpreted bits.
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, 0.03, 200, 0, 255, torch.uint8
        ).to(torch.int32)


@register_test_case(
    module_factory=lambda: QuantizedDecomposedQuantizePerTensorUnsigned()
)
def QuantizedDecomposedQuantizePerTensorUnsigned_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(4, 8) - 5)


# ==============================================================================


class QuantizedDecomposedQuantizePerTensorTensor(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
            ([], torch.float32, True),
            ([], torch.int32, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            x, scale, zero_point, -128, 127, torch.int8
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedQuantizePerTensorTensor())
def QuantizedDecomposedQuantizePerTensorTensor_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(4, 8),
        torch.tensor(0.03, dtype=torch.float32),
        torch.tensor(-10, dtype=torch.int32),
    )


# ==============================================================================


class QuantizedDecomposedQuantizePerTensorTensor2(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
            ([], torch.float32, True),
            ([], torch.int32, True),
            ([], torch.int32, True),
            ([], torch.int32, True),
        ]
    )
    def forward(self, x, scale, zero_point, qmin, qmax):
        return torch.ops.quantized_decomposed.quantize_per_tensor.tensor2(
            x, scale, zero_point, qmin, qmax, torch.int8
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedQuantizePerTensorTensor2()
)
def QuantizedDecomposedQuantizePerTensorTensor2_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(4, 8),
        torch.tensor(0.03, dtype=torch.float32),
        torch.tensor(-10, dtype=torch.int32),
        torch.tensor(-128, dtype=torch.int32),
        torch.tensor(127, dtype=torch.int32),
    )


# ==============================================================================


class QuantizedDecomposedDequantizePerTensorTensor(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.int8, True),
            ([], torch.float32, True),
            ([], torch.int32, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            x, scale, zero_point, -128, 127, torch.int8
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerTensorTensor()
)
def QuantizedDecomposedDequantizePerTensorTensor_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 8, low=-128, high=127).to(torch.int8),
        torch.tensor(0.03, dtype=torch.float32),
        torch.tensor(-10, dtype=torch.int32),
    )


# ==============================================================================


class QuantizedDecomposedDequantizePerTensorTensor2(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.int8, True),
            ([], torch.float32, True),
            ([], torch.int32, True),
            ([], torch.int32, True),
            ([], torch.int32, True),
        ]
    )
    def forward(self, x, scale, zero_point, qmin, qmax):
        return torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2(
            x, scale, zero_point, qmin, qmax, torch.int8
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerTensorTensor2()
)
def QuantizedDecomposedDequantizePerTensorTensor2_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 8, low=-128, high=127).to(torch.int8),
        torch.tensor(0.03, dtype=torch.float32),
        torch.tensor(-10, dtype=torch.int32),
        torch.tensor(-128, dtype=torch.int32),
        torch.tensor(127, dtype=torch.int32),
    )


# ==============================================================================


class QuantizedDecomposedChooseQparamsTensor(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
            x, -128, 127, 1e-8, torch.int8
        )
        return scale, zp


@register_test_case(module_factory=lambda: QuantizedDecomposedChooseQparamsTensor())
def QuantizedDecomposedChooseQparamsTensor_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 8))


# ==============================================================================


class QuantizedDecomposedChooseQparamsSymmetricTensor(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor(
            x, -128, 127, 1e-8, torch.int8
        )
        return scale, zp


@register_test_case(
    module_factory=lambda: QuantizedDecomposedChooseQparamsSymmetricTensor()
)
def QuantizedDecomposedChooseQparamsSymmetricTensor_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 8))


# ==============================================================================


class QuantizedDecomposedChooseQparamsSymmetricTensorUint8(torch.nn.Module):
    """Exercise the zp=128 path: dtype=torch.uint8 triggers the unsigned 8-bit
    branch in the symmetric choose_qparams lowering."""

    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor(
            x, 0, 255, 1e-8, torch.uint8
        )
        return scale, zp


@register_test_case(
    module_factory=lambda: QuantizedDecomposedChooseQparamsSymmetricTensorUint8()
)
def QuantizedDecomposedChooseQparamsSymmetricTensorUint8_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 8))


# ==============================================================================


class QuantizedDecomposedDynamicQuantFlowSymmetric(torch.nn.Module):
    """Symmetric dynamic quantization flow with tensor2:
    choose_qparams_symmetric -> quantize_per_tensor.tensor2 -> dequantize_per_tensor.tensor2
    """

    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor(
            x, -128, 127, 1e-8, torch.int8
        )
        qmin = torch.tensor(-128, dtype=torch.int32)
        qmax = torch.tensor(127, dtype=torch.int32)
        xq = torch.ops.quantized_decomposed.quantize_per_tensor.tensor2(
            x, scale, zp, qmin, qmax, torch.int8
        )
        xdq = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2(
            xq, scale, zp, qmin, qmax, torch.int8
        )
        return xdq


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDynamicQuantFlowSymmetric()
)
def QuantizedDecomposedDynamicQuantFlowSymmetric_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 8))


# ==============================================================================


class QuantizedDecomposedDynamicQuantFlowAsymmetric(torch.nn.Module):
    """Asymmetric dynamic quantization flow:
    choose_qparams -> quantize_per_tensor -> dequantize_per_tensor.
    """

    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        scale, zp = torch.ops.quantized_decomposed.choose_qparams.tensor(
            x, -128, 127, 1e-8, torch.int8
        )
        xq = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
            x, scale, zp, -128, 127, torch.int8
        )
        xdq = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(
            xq, scale, zp, -128, 127, torch.int8
        )
        return xdq


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDynamicQuantFlowAsymmetric()
)
def QuantizedDecomposedDynamicQuantFlowAsymmetric_basic(module, tu: TestUtils):
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


class QuantizedDecomposedDequantizePerChannelUnsignedZeroPoint(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.uint8, True),
            ([8], torch.float32, True),
            ([8], torch.uint8, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.dequantize_per_channel.default(
            x, scales, zero_points, 1, 0, 255, torch.uint8
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerChannelUnsignedZeroPoint()
)
def QuantizedDecomposedDequantizePerChannelUnsignedZeroPoint_basic(
    module, tu: TestUtils
):
    # The eager reference subtracts the zero points in uint8, which wraps when
    # an input is below its zero point, so the inputs stay above every zero
    # point. Zero points >= 128 are the ones a sign extension would corrupt.
    module.forward(
        tu.randint(4, 8, low=192, high=256).to(torch.uint8),
        tu.rand(8) + 0.01,
        tu.randint(8, low=128, high=192).to(torch.uint8),
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


class QuantizedDecomposedDequantizePerChannelGroup(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.int8, True),
            ([4, 4], torch.float32, True),
            ([4, 4], torch.int64, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
            x, scales, zero_points, -128, 127, torch.int8, 4, torch.float32
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerChannelGroup()
)
def QuantizedDecomposedDequantizePerChannelGroup_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 16, low=-128, high=127).to(torch.int8),
        tu.rand(4, 4) + 0.01,
        tu.randint(4, 4, low=-128, high=127).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedDequantizePerChannelGroupUnsignedSymmetric(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([8, 32], torch.uint8, True),
            ([8, 4], torch.float32, True),
        ]
    )
    def forward(self, x, scales):
        return torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
            x, scales, None, 0, 255, torch.uint8, 8, torch.float32
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedDequantizePerChannelGroupUnsignedSymmetric()
)
def QuantizedDecomposedDequantizePerChannelGroupUnsignedSymmetric_basic(
    module, tu: TestUtils
):
    module.forward(
        tu.randint(8, 32, low=128, high=255).to(torch.uint8),
        tu.rand(8, 4) + 0.01,
    )


# ==============================================================================


class QuantizedDecomposedQuantizePerChannelGroup(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.float32, True),
            ([4, 4], torch.float32, True),
            ([4, 4], torch.int8, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        return torch.ops.quantized_decomposed.quantize_per_channel_group.default(
            x, scales, zero_points, -128, 127, torch.int8, 4
        )


@register_test_case(module_factory=lambda: QuantizedDecomposedQuantizePerChannelGroup())
def QuantizedDecomposedQuantizePerChannelGroup_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(4, 16) - 5,
        tu.rand(4, 4) + 0.01,
        tu.randint(4, 4, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================


class QuantizedDecomposedPerChannelGroupGptqSingleCol(torch.nn.Module):
    """GPTQ single-column quantize -> dequantize: group_size 128 must
    be clamped to 16."""

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
        q = torch.ops.quantized_decomposed.quantize_per_channel_group.default(
            x, scales, zero_points, -128, 127, torch.int8, 128
        )
        return torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
            q, scales, zero_points, -128, 127, torch.int8, 128, torch.float32
        )


@register_test_case(
    module_factory=lambda: QuantizedDecomposedPerChannelGroupGptqSingleCol()
)
def QuantizedDecomposedPerChannelGroupGptqSingleCol_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(4, 16) - 5,
        tu.rand(4, 1) + 0.01,
        tu.randint(4, 1, low=-10, high=10).to(torch.int64),
    )


# ==============================================================================


class QuantizedDecomposedQuantizePerChannelUnsignedZeroPoint(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 8], torch.float32, True),
            ([8], torch.float32, True),
            ([8], torch.uint8, True),
        ]
    )
    def forward(self, x, scales, zero_points):
        # The refbackend returns every i8 buffer as int8, so widen the uint8
        # result to compare its values rather than its reinterpreted bits.
        return torch.ops.quantized_decomposed.quantize_per_channel.default(
            x, scales, zero_points, 1, 0, 255, torch.uint8
        ).to(torch.int32)


@register_test_case(
    module_factory=lambda: QuantizedDecomposedQuantizePerChannelUnsignedZeroPoint()
)
def QuantizedDecomposedQuantizePerChannelUnsignedZeroPoint_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(4, 8) - 5,
        tu.rand(8) + 0.01,
        tu.randint(8, low=128, high=256).to(torch.uint8),
    )
