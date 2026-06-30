# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.annotations import annotate_args, export
from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case

# ==============================================================================


class AtenDotModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.dot(lhs, rhs)


@register_test_case(module_factory=lambda: AtenDotModule())
def AtenDotModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4))


# ==============================================================================


class MatmulDot(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulDot())
def Matmul_dot(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(3))


# ==============================================================================


class Matmul2D(torch.nn.Module):
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
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul2D())
def Matmul_2d(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(4, 5))


# ==============================================================================


class MatmulZeroK(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 0], torch.float32, True),
            ([0, 10], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulZeroK())
def MatmulZeroK_basic(module, tu: TestUtils):
    module.forward(torch.empty(5, 0), torch.empty(0, 10))


class AtenScaledMmPerTensorModule(torch.nn.Module):
    def __init__(self, fp8_dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16):
        super().__init__()
        self.fp8_dtype = fp8_dtype
        self.out_dtype = out_dtype

    @export
    @annotate_args(
        [
            None,
            ([16, 16], torch.float32, True),
            ([16, 16], torch.float32, True),
            ([], torch.float32, True),
            ([], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs, scale_lhs, scale_rhs):
        lhs_fp8 = lhs.to(self.fp8_dtype)
        rhs_fp8 = rhs.to(self.fp8_dtype)
        return torch._scaled_mm(
            lhs_fp8,
            rhs_fp8,
            scale_lhs,
            scale_rhs,
            out_dtype=self.out_dtype,
        )


@register_test_case(module_factory=lambda: AtenScaledMmPerTensorModule())
def AtenScaledMmPerTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(16, 16), tu.rand(16, 16), tu.rand(), tu.rand())


@register_test_case(
    module_factory=lambda: AtenScaledMmPerTensorModule(out_dtype=torch.float16)
)
def AtenScaledMmPerTensorF16Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(16, 16), tu.rand(16, 16), tu.rand(), tu.rand())


@register_test_case(
    module_factory=lambda: AtenScaledMmPerTensorModule(out_dtype=torch.float32)
)
def AtenScaledMmPerTensorF32Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(16, 16), tu.rand(16, 16), tu.rand(), tu.rand())


@register_test_case(
    module_factory=lambda: AtenScaledMmPerTensorModule(fp8_dtype=torch.float8_e5m2)
)
def AtenScaledMmPerTensorE5M2Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(16, 16), tu.rand(16, 16), tu.rand(), tu.rand())


# ==============================================================================


class AtenScaledMmBlockScaledFp8Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 32], torch.float32, True),
            ([32, 64], torch.float32, True),
            ([512], torch.float8_e8m0fnu, True),
            ([512], torch.float8_e8m0fnu, True),
        ]
    )
    def forward(self, lhs, rhs, scale_lhs, scale_rhs):
        lhs_fp8 = lhs.to(torch.float8_e4m3fn)
        rhs_fp8 = rhs.to(torch.float8_e4m3fn)
        return torch._scaled_mm(
            lhs_fp8,
            rhs_fp8,
            scale_lhs,
            scale_rhs,
            out_dtype=torch.bfloat16,
        )


@register_test_case(module_factory=lambda: AtenScaledMmBlockScaledFp8Module())
def AtenScaledMmBlockScaledFp8Module_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(3, 32),
        tu.rand(32, 64),
        torch.ones(512, dtype=torch.float8_e8m0fnu),
        torch.ones(512, dtype=torch.float8_e8m0fnu),
    )


class AtenScaledMmBlockScaledFp8SwizzledModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "scale_lhs", torch.ones(32, 16, dtype=torch.float8_e8m0fnu)
        )
        self.register_buffer(
            "scale_rhs", torch.ones(32, 16, dtype=torch.float8_e8m0fnu)
        )

    @export
    @annotate_args(
        [
            None,
            ([3, 32], torch.float32, True),
            ([32, 64], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        lhs_fp8 = lhs.to(torch.float8_e4m3fn)
        rhs_fp8 = rhs.to(torch.float8_e4m3fn)
        return torch._scaled_mm(
            lhs_fp8,
            rhs_fp8,
            self.scale_lhs,
            self.scale_rhs,
            out_dtype=torch.bfloat16,
        )


@register_test_case(module_factory=lambda: AtenScaledMmBlockScaledFp8SwizzledModule())
def AtenScaledMmBlockScaledFp8SwizzledModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 32), tu.rand(32, 64))


# ==============================================================================


class MatmulVecMat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulVecMat())
def Matmul_vecmat(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4, 5))


# ==============================================================================


class MatmulMatVec(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulMatVec())
def Matmul_matvec(module, tu: TestUtils):
    module.forward(tu.rand(4, 5), tu.rand(5))


# ==============================================================================


class Matmul3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul3D())
def Matmul_3d(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 5, 4))


# ==============================================================================


class Matmul3DStaticBroadcast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 8, 5], torch.float32, True),
            ([1, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul3DStaticBroadcast())
def Matmul3DStaticBroadcast_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 8, 5), tu.rand(1, 5, 6))


# ==============================================================================


class Matmul4d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul4d())
def Matmul_4d(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(4, 5, 7, 6))


# ==============================================================================


class Matmul4dStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 5, 6, 7], torch.float32, True),
            ([4, 5, 7, 6], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul4dStatic())
def Matmul4dStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(4, 5, 7, 6))


# ==============================================================================


class MatmulStaticBroadcast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 1, 6, 7], torch.float32, True),
            ([8, 1, 5, 7, 6], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulStaticBroadcast())
def MatmulStaticBroadcast_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 1, 6, 7), tu.rand(8, 1, 5, 7, 6))


# ==============================================================================


class MatmulSingleDynamicBatchDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, -1, -1, -1], torch.float32, True),
            ([4, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulSingleDynamicBatchDim())
def MatmulSingleDynamicBatchDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(4, 5, 7, 6))


# ==============================================================================


class MatmulBroadcastBatchDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulBroadcastBatchDim())
def MatmulBroadcastBatchDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(5, 7, 6))


# ==============================================================================


class Mv(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, m, v):
        return torch.mv(m, v)


@register_test_case(module_factory=lambda: Mv())
def Mv_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2), tu.rand(2))


# ==============================================================================


class AtenMmFloatTypes(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.mm(a, b)


@register_test_case(module_factory=lambda: AtenMmFloatTypes())
def AtenMmFloatTypes_basic(module, tu: TestUtils):
    module.forward(tu.rand(8, 8), tu.rand(8, 8))


# ==============================================================================


class AtenMmZeroK(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([5, 0], torch.float32, True),
            ([0, 10], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.mm(a, b)


@register_test_case(module_factory=lambda: AtenMmZeroK())
def AtenMmZeroK_basic(module, tu: TestUtils):
    module.forward(torch.empty(5, 0), torch.empty(0, 10))


# ==============================================================================


class AtenBmmZeroK(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 5, 0], torch.float32, True),
            ([2, 0, 10], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.bmm(a, b)


@register_test_case(module_factory=lambda: AtenBmmZeroK())
def AtenBmmZeroK_basic(module, tu: TestUtils):
    module.forward(torch.empty(2, 5, 0), torch.empty(2, 0, 10))


# ==============================================================================


class AtenMmIntTypes(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.mm(a, b)


@register_test_case(module_factory=lambda: AtenMmIntTypes())
def AtenMmIntTypes_basic(module, tu: TestUtils):
    module.forward(tu.randint(16, 4, high=100), tu.randint(4, 16, high=100))


# ==============================================================================


class AtenMmInt8Types(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int8, True),
            ([-1, -1], torch.int8, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.mm(a, b)


@register_test_case(module_factory=lambda: AtenMmInt8Types())
def AtenMmInt8Types_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(16, 4, high=100).to(torch.int8),
        tu.randint(4, 16, high=100).to(torch.int8),
    )


# ==============================================================================


class AtenMmInt8ZeroK(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3, 0], torch.int8, True),
            ([0, 3], torch.int8, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.mm(a, b)


@register_test_case(module_factory=lambda: AtenMmInt8ZeroK())
def AtenMmInt8ZeroK_basic(module, tu: TestUtils):
    module.forward(
        torch.empty(3, 0, dtype=torch.int8),
        torch.empty(0, 3, dtype=torch.int8),
    )


# ==============================================================================


class AtenMmF16Types(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float16, True),
            ([-1, -1], torch.float16, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.mm(a, b)


@register_test_case(module_factory=lambda: AtenMmF16Types())
def AtenMmF16Types_basic(module, tu: TestUtils):
    module.forward(tu.rand(16, 4).to(torch.float16), tu.rand(4, 16).to(torch.float16))


# ==============================================================================
# For DQ-Q fake quantization ops
import torch.ao.quantization.fx._decomposed


class AtenMmQint8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.int8, True),
            ([4, 3], torch.int8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.0215, -25, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.0176, 18, -128, 127, torch.int8
        )
        z = torch.mm(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMmQint8())
def AtenMmQint8_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(4, 3, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================


class AtenMmQuint8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.uint8, True),
            ([4, 3], torch.uint8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.199, 65, 0, 255, torch.uint8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.0215, 160, 0, 255, torch.uint8
        )
        z = torch.mm(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMmQuint8())
def AtenMmQuint8_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=0, high=255).to(torch.uint8),
        tu.randint(4, 3, low=0, high=255).to(torch.uint8),
    )


# ==============================================================================


class AtenMmQMixedSigni8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.int8, True),
            ([4, 3], torch.uint8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.03, -66, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.025, 160, 0, 255, torch.uint8
        )
        z = torch.mm(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMmQMixedSigni8())
def AtenMmQMixedSigni8_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(4, 3, low=0, high=255).to(torch.uint8),
    )


# ==============================================================================


class AtenIntMM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.int8, True),
            ([4, 3], torch.int8, True),
        ]
    )
    def forward(self, x, y):
        return torch._int_mm(x, y)


@register_test_case(module_factory=lambda: AtenIntMM())
def AtenIntMM_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(4, 3, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================


class AtenMatmulQint8VM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.int8, True),
            ([-1, -1], torch.int8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.0215, -25, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.0176, 18, -128, 127, torch.int8
        )
        z = torch.matmul(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMatmulQint8VM())
def AtenMatmulQint8VM_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(9, low=-128, high=127).to(torch.int8),
        tu.randint(9, 4, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================
class AtenMatmulQint8VV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.int8, True),
            ([-1], torch.int8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.0215, -25, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.0176, 18, -128, 127, torch.int8
        )
        z = torch.matmul(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMatmulQint8VV())
def AtenMatmulQint8VV_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(9, low=-128, high=127).to(torch.int8),
        tu.randint(9, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================
class AtenMatmulQint8MV(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int8, True),
            ([-1], torch.int8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.0215, -25, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.0176, 18, -128, 127, torch.int8
        )
        z = torch.matmul(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMatmulQint8MV())
def AtenMatmulQint8MV_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(4, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================
class AtenMatmulQint8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, -1, 3, 4], torch.int8, True),
            ([-1, 4, 3], torch.int8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.0215, -25, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.0176, 18, -128, 127, torch.int8
        )
        z = torch.matmul(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMatmulQint8())
def AtenMatmulQint8_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 7, 3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(7, 4, 3, low=-128, high=127).to(torch.int8),
    )


# ==============================================================================


class AtenMatmulQMixedSigni8(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([7, -1, -1, -1], torch.int8, True),
            ([-1, -1, -1], torch.uint8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.03, -66, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.025, 160, 0, 255, torch.uint8
        )
        z = torch.matmul(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMatmulQMixedSigni8())
def AtenMatmulQMixedSigni8_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(7, 2, 3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(2, 4, 3, low=0, high=255).to(torch.uint8),
    )


# ==============================================================================


class AtenMatmulQMixedSigni8Transpose(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([7, -1, -1, -1], torch.int8, True),
            ([-1, -1, -1], torch.uint8, True),
        ]
    )
    def forward(self, x, y):
        x = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, 0.03, -66, -128, 127, torch.int8
        )
        y = torch.torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, 0.025, 160, 0, 255, torch.uint8
        )
        y = torch.transpose(y, 1, 2)
        z = torch.matmul(x, y)
        return z


@register_test_case(module_factory=lambda: AtenMatmulQMixedSigni8Transpose())
def AtenMatmulQMixedSigni8Transpose_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(7, 2, 3, 4, low=-128, high=127).to(torch.int8),
        tu.randint(2, 6, 4, low=0, high=255).to(torch.uint8),
    )


# ==============================================================================


class AtenLinear1D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3], torch.float32, True),
            ([3], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linear(a, b)


@register_test_case(module_factory=lambda: AtenLinear1D())
def AtenLinear1D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(3))


# ==============================================================================


class AtenLinearMatVec(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.float32, True),
            ([4], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linear(a, b)


@register_test_case(module_factory=lambda: AtenLinearMatVec())
def AtenLinearMatVec_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(4))


# ==============================================================================


class AtenLinearVecMat(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4], torch.float32, True),
            ([3, 4], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linear(a, b)


@register_test_case(module_factory=lambda: AtenLinearVecMat())
def AtenLinearVecMat_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(3, 4))


class AtenLinearVecMatBias(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4], torch.float32, True),
            ([3, 4], torch.float32, True),
            ([3], torch.float32, True),
        ]
    )
    def forward(self, a, b, c):
        return torch.ops.aten.linear(a, b, c)


@register_test_case(module_factory=lambda: AtenLinearVecMatBias())
def AtenLinearVecMatBias_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(3, 4), tu.rand(3))


# ==============================================================================


class AtenLinear2D(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.float32, True),
            ([5, 4], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linear(a, b)


@register_test_case(module_factory=lambda: AtenLinear2D())
def AtenLinear2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(5, 4))


# ==============================================================================


class AtenLinear3DBias(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([3, 6, 4], torch.float32, True),
            ([5, 4], torch.float32, True),
            ([5], torch.float32, True),
        ]
    )
    def forward(self, a, b, c):
        return torch.ops.aten.linear(a, b, c)


@register_test_case(module_factory=lambda: AtenLinear3DBias())
def AtenLinear3DBias_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 6, 4), tu.rand(5, 4), tu.rand(5))


# ==============================================================================


class AtenLinalgCrossInt(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 3], torch.int64, True),
            ([2, 3], torch.int64, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linalg_cross(a, b)


@register_test_case(module_factory=lambda: AtenLinalgCrossInt())
def AtenLinalgCrossInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3), tu.randint(2, 3))


# ==============================================================================


class AtenLinalgCrossFloat(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 3], torch.float32, True),
            ([2, 3], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linalg_cross(a, b)


@register_test_case(module_factory=lambda: AtenLinalgCrossFloat())
def AtenLinalgCrossFloat_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3), tu.rand(2, 3))


# ==============================================================================


class AtenLinalgCrossBroadcast(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([1, 4, 3], torch.float32, True),
            ([5, 4, 3], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linalg_cross(a, b)


@register_test_case(module_factory=lambda: AtenLinalgCrossBroadcast())
def AtenLinalgCrossBroadcast_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 3), tu.rand(5, 4, 3))


# ==============================================================================


class AtenLinalgCrossCustomDim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([1, 4, 3, 2, 2], torch.float32, True),
            ([5, 4, 3, 2, 1], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linalg_cross(a, b, dim=2)


@register_test_case(module_factory=lambda: AtenLinalgCrossCustomDim())
def AtenLinalgCrossCustomDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 3, 2, 2), tu.rand(5, 4, 3, 2, 1))


# ==============================================================================


class AtenLinalgCrossNegativeDim(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([1, 4, 3, 2, 2], torch.float32, True),
            ([5, 4, 3, 2, 1], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linalg_cross(a, b, dim=-3)


@register_test_case(module_factory=lambda: AtenLinalgCrossNegativeDim())
def AtenLinalgCrossNegativeDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 3, 2, 2), tu.rand(5, 4, 3, 2, 1))


# ==============================================================================


class AtenLinalgCrossDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a, b):
        return torch.ops.aten.linalg_cross(a, b, dim=1)


@register_test_case(module_factory=lambda: AtenLinalgCrossDynamic())
def AtenLinalgCrossDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 3, 1, 6), tu.rand(4, 3, 7, 1))


# ==============================================================================


class AtenOuterFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.float32, True), ([-1], torch.float32, True)])
    def forward(self, a, b):
        return torch.ops.aten.outer(a, b)


@register_test_case(module_factory=lambda: AtenOuterFloat())
def AtenOuterFloat_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(3))


# ==============================================================================


class AtenOuterInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.int64, True), ([-1], torch.int64, True)])
    def forward(self, a, b):
        return torch.ops.aten.outer(a, b)


@register_test_case(module_factory=lambda: AtenOuterInt())
def AtenOuterInt_basic(module, tu: TestUtils):
    module.forward(tu.randint(4), tu.randint(3))


# ==============================================================================


class AtenOuterF32F64(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.float32, True), ([-1], torch.float64, True)])
    def forward(self, a, b):
        return torch.ops.aten.outer(a, b)


@register_test_case(module_factory=lambda: AtenOuterF32F64())
def AtenOuterF32F64_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(3).to(torch.float64))
