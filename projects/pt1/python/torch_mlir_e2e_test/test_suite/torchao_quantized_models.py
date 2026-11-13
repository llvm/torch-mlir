# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerAxis, PerGroup

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class TorchaoQuantizeAffinePerToken(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 16, 64], torch.float32, True),
            ([2, 16, 1], torch.float32, True),
            ([2, 16, 1], torch.int8, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.torchao.quantize_affine.default(
            x, [1, 1, 64], scale, zero_point, torch.int8
        )


@register_test_case(module_factory=lambda: TorchaoQuantizeAffinePerToken())
def TorchaoQuantizeAffinePerToken_basic(module, tu: TestUtils):
    module.forward(
        10 * tu.rand(2, 16, 64) - 5,
        tu.rand(2, 16, 1) + 0.01,
        tu.randint(2, 16, 1, low=-10, high=10).to(torch.int8),
    )


# ==============================================================================


class TorchaoDequantizeAffinePerGroup(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.int8, True),
            ([4, 2], torch.float32, True),
            ([4, 2], torch.float32, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        # NumPy has no bf16 type, so the scale is passed as f32 and cast here.
        return torch.ops.torchao.dequantize_affine.default(
            x, [1, 8], scale.to(torch.bfloat16), zero_point, torch.int8, -128, 127
        )


@register_test_case(module_factory=lambda: TorchaoDequantizeAffinePerGroup())
def TorchaoDequantizeAffinePerGroup_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 16, low=-128, high=128).to(torch.int8),
        tu.rand(4, 2) + 0.01,
        20 * tu.rand(4, 2) - 10,
    )


# ==============================================================================


class TorchaoChooseQparamsAffineSymmetricPerTensorKeepdim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 16, 64], torch.float32, True),
        ]
    )
    def forward(self, x):
        # The per-tensor activation call of Int8DynamicActivationInt8WeightConfig.
        return torch.ops.torchao.choose_qparams_affine.default(
            x,
            "SYMMETRIC",
            [2, 16, 64],
            torch.int8,
            -128,
            127,
            torch.finfo(torch.float32).eps,
            torch.float32,
            torch.int8,
            keepdim=True,
        )


@register_test_case(
    module_factory=lambda: TorchaoChooseQparamsAffineSymmetricPerTensorKeepdim()
)
def TorchaoChooseQparamsAffineSymmetricPerTensorKeepdim_basic(module, tu: TestUtils):
    module.forward(2 * tu.rand(2, 16, 64) - 1)


# ==============================================================================


class TorchaoChooseQparamsAffineSymmetricNoClippingErrPerGroupKeepdim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([32, 64], torch.float32, True),
        ]
    )
    def forward(self, x):
        # The int4 per-group weight call of IntxWeightOnlyConfig.
        return torch.ops.torchao.choose_qparams_affine.default(
            x,
            "SYMMETRIC_NO_CLIPPING_ERR",
            [1, 32],
            torch.int8,
            -8,
            7,
            zero_point_dtype=torch.int8,
            keepdim=True,
        )


@register_test_case(
    module_factory=lambda: TorchaoChooseQparamsAffineSymmetricNoClippingErrPerGroupKeepdim()
)
def TorchaoChooseQparamsAffineSymmetricNoClippingErrPerGroupKeepdim_basic(
    module, tu: TestUtils
):
    module.forward(6 * tu.rand(32, 64) - 2)


# ==============================================================================


class TorchaoQuantizeRoundTripAsymmetricPerToken(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 16, 64], torch.float32, True),
        ]
    )
    def forward(self, x):
        # The activation quantization of Int8DynamicActivationIntxWeightConfig.
        block_size = [1, 1, 64]
        scale, zero_point = torch.ops.torchao.choose_qparams_affine.default(
            x, "ASYMMETRIC", block_size, torch.int8, -128, 127, None, None, torch.int8
        )
        quantized = torch.ops.torchao.quantize_affine.default(
            x, block_size, scale, zero_point, torch.int8, -128, 127
        )
        dequantized = torch.ops.torchao.dequantize_affine.default(
            quantized, block_size, scale, zero_point, torch.int8, -128, 127
        )
        return scale, zero_point, dequantized


@register_test_case(module_factory=lambda: TorchaoQuantizeRoundTripAsymmetricPerToken())
def TorchaoQuantizeRoundTripAsymmetricPerToken_basic(module, tu: TestUtils):
    module.forward(10 * tu.rand(2, 16, 64) - 3)


# ==============================================================================


class TorchaoQuantizeRoundTripSymmetricPerTokenKeepdim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 16, 64], torch.float32, True),
        ]
    )
    def forward(self, x):
        # The per-token activation call of Int8DynamicActivationInt8WeightConfig.
        block_size = [1, 1, 64]
        scale, zero_point = torch.ops.torchao.choose_qparams_affine.default(
            x,
            "SYMMETRIC",
            block_size,
            torch.int8,
            -128,
            127,
            torch.finfo(torch.float32).eps,
            torch.float32,
            torch.int8,
            keepdim=True,
        )
        quantized = torch.ops.torchao.quantize_affine.default(
            x, block_size, scale, zero_point, torch.int8, -128, 127
        )
        dequantized = torch.ops.torchao.dequantize_affine.default(
            quantized, block_size, scale, zero_point, torch.int8, -128, 127
        )
        return scale, zero_point, dequantized


@register_test_case(
    module_factory=lambda: TorchaoQuantizeRoundTripSymmetricPerTokenKeepdim()
)
def TorchaoQuantizeRoundTripSymmetricPerTokenKeepdim_basic(module, tu: TestUtils):
    x = tu.randint(2, 16, 64, low=-127, high=128).to(torch.float32)
    x[:, :, 0] = 127.5
    x[:, 1::2, 0] = -127.5
    module.forward(x / 4)


# ==============================================================================


class TorchaoLinearInt8DynamicActivationInt4Weight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.linear = torch.nn.Linear(64, 32)
        quantize_(
            self,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(32)
            ),
        )

    @export
    @annotate_args(
        [
            None,
            ([2, 8, 64], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.linear(x)


# torchao stores the int4 weights in int8 and dequantizes them with the int4
# range as quant_min and quant_max, so sub-byte weights are covered next to the
# per-token quantization of the activations.
@register_test_case(
    module_factory=lambda: TorchaoLinearInt8DynamicActivationInt4Weight()
)
def TorchaoLinearInt8DynamicActivationInt4Weight_basic(module, tu: TestUtils):
    module.forward(2 * tu.rand(2, 8, 64) - 1)


# ==============================================================================


class TorchaoLinearInt4WeightOnlyPerAxisF16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(0)
        self.linear = torch.nn.Linear(64, 32, dtype=torch.float16)
        quantize_(
            self,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0)),
        )

    @export
    @annotate_args(
        [
            None,
            ([2, 8, 64], torch.float16, True),
        ]
    )
    def forward(self, x):
        return self.linear(x)


# An f16 model gets f16 scales, which the lowering extends to f32 before it
# truncates the result back to f16. bf16 models are more common, but the
# refbackend cannot compile a bf16 Linear.
@register_test_case(module_factory=lambda: TorchaoLinearInt4WeightOnlyPerAxisF16())
def TorchaoLinearInt4WeightOnlyPerAxisF16_basic(module, tu: TestUtils):
    module.forward((2 * tu.rand(2, 8, 64) - 1).to(torch.float16))
