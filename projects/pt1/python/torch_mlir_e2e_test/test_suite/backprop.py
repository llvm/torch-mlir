# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class SoftmaxBackwardModule(torch.nn.Module):
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
    def forward(self, grad_output, output):
        return torch.ops.aten._softmax_backward_data(
            grad_output, output, dim=1, input_dtype=6
        )


@register_test_case(module_factory=lambda: SoftmaxBackwardModule())
def SoftmaxBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))


# ==============================================================================
class TanhBackwardModule(torch.nn.Module):
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
    def forward(self, grad_out, output):
        return torch.ops.aten.tanh_backward(grad_out, output)


@register_test_case(module_factory=lambda: TanhBackwardModule())
def TanhBackward_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3), tu.rand(3, 3))


# ==============================================================================


class HardtanhBackwardModule(torch.nn.Module):
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
    def forward(self, grad_out, input):
        return torch.ops.aten.hardtanh_backward(
            grad_out, input, min_val=0.2, max_val=0.5
        )


@register_test_case(module_factory=lambda: HardtanhBackwardModule())
def HardtanhBackward_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 20), tu.rand(10, 20))


# ==============================================================================


class ConvolutionBackwardModule2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=None,
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0],
            groups=1,
            output_mask=[True, True, True],
        )


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2D())
def ConvolutionBackwardModule2D_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(tu.rand(2, 2, 5, 5), tu.rand(2, 2, 6, 6), tu.rand(2, 2, 2, 2))


class ConvolutionBackwardModule2DStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 4, 64, 64], torch.float32, True),
            ([1, 320, 64, 64], torch.float32, True),
            ([4, 320, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=[4],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
            output_mask=[True, True, True],
        )


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2DStatic())
def ConvolutionBackwardModule2DStatic_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(
            tu.rand(1, 4, 64, 64), tu.rand(1, 320, 64, 64), tu.rand(4, 320, 3, 3)
        )


class ConvolutionBackwardModule3DStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 4, 64, 64, 64], torch.float32, True),
            ([1, 320, 64, 64, 64], torch.float32, True),
            ([4, 320, 3, 1, 3], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=[4],
            stride=[1, 1, 1],
            padding=[1, 0, 1],
            dilation=[1, 1, 1],
            transposed=False,
            output_padding=[0, 0, 0],
            groups=1,
            output_mask=[True, True, True],
        )


@register_test_case(module_factory=lambda: ConvolutionBackwardModule3DStatic())
def ConvolutionBackwardModule3DStatic_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(
            tu.rand(1, 4, 64, 64, 64),
            tu.rand(1, 320, 64, 64, 64),
            tu.rand(4, 320, 3, 1, 3),
        )


class ConvolutionBackwardModule2DPadded(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=None,
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0],
            groups=1,
            output_mask=[True, True, True],
        )


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2DPadded())
def ConvolutionBackwardModule2DPadded_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(tu.rand(2, 2, 8, 8), tu.rand(2, 2, 6, 6), tu.rand(2, 2, 3, 3))


class ConvolutionBackwardModule2DStrided(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 4, 4], torch.float32, True),
            ([1, 2, 8, 8], torch.float32, True),
            ([2, 2, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=[4],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
            output_mask=[True, True, True],
        )


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2DStrided())
def ConvolutionBackwardModule2DStrided_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(tu.rand(1, 2, 4, 4), tu.rand(1, 2, 8, 8), tu.rand(2, 2, 3, 3))


class ConvolutionBackwardModule2DDilated(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 6, 6], torch.float32, True),
            ([1, 4, 8, 8], torch.float32, True),
            ([2, 4, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=[4],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[2, 2],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
            output_mask=[True, True, True],
        )


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2DDilated())
def ConvolutionBackwardModule2DDilated_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(tu.rand(1, 2, 6, 6), tu.rand(1, 4, 8, 8), tu.rand(2, 4, 3, 3))


class ConvolutionBackwardModule2DStridedPaddedDilatedGrouped(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 16, 32, 32], torch.float32, True),
            ([2, 128, 64, 64], torch.float32, True),
            ([16, 32, 2, 2], torch.float32, True),
        ]
    )
    def forward(self, grad_out, input_vec, weight):
        return torch.ops.aten.convolution_backward(
            grad_out,
            input_vec,
            weight,
            bias_sizes=[4],
            stride=[2, 2],
            padding=[2, 2],
            dilation=[4, 4],
            transposed=False,
            output_padding=[0, 0],
            groups=4,
            output_mask=[True, True, True],
        )


@register_test_case(
    module_factory=lambda: ConvolutionBackwardModule2DStridedPaddedDilatedGrouped()
)
def ConvolutionBackwardModule2DStridedPaddedDilatedGrouped_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(
            tu.rand(2, 16, 32, 32), tu.rand(2, 128, 64, 64), tu.rand(16, 32, 2, 2)
        )


# ==============================================================================


class GeluBackwardModule(torch.nn.Module):
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
    def forward(self, grad, input):
        return torch.ops.aten.gelu_backward(grad, input)


@register_test_case(module_factory=lambda: GeluBackwardModule())
def GeluBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3), tu.rand(5, 3))


class LogSoftmaxBackwardModule(torch.nn.Module):
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
    def forward(self, grad_output, output):
        return torch.ops.aten._log_softmax_backward_data(
            grad_output, output, dim=1, input_dtype=6
        )


@register_test_case(module_factory=lambda: LogSoftmaxBackwardModule())
def LogSoftmaxBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))


# ==============================================================================


class LeakyReluBackwardModule(torch.nn.Module):
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
    def forward(self, grad, input):
        return torch.ops.aten.leaky_relu_backward(
            grad, input, negative_slope=0.1, self_is_result=False
        )


@register_test_case(module_factory=lambda: LeakyReluBackwardModule())
def LeakyReluBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3), tu.rand(5, 3))


class LeakyReluBackwardStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5], torch.float32, True),
            ([3, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, grad, input):
        return torch.ops.aten.leaky_relu_backward(
            grad, input, negative_slope=0.1, self_is_result=False
        )


@register_test_case(module_factory=lambda: LeakyReluBackwardStaticModule())
def LeakyReluBackwardStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 5))


# ==============================================================================


class RreluWithNoiseBackwardTrainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, grad, input, noise):
        return torch.ops.aten.rrelu_with_noise_backward(
            grad,
            input,
            noise,
            lower=0.1,
            upper=0.9,
            training=True,
            self_is_result=False,
        )


@register_test_case(module_factory=lambda: RreluWithNoiseBackwardTrainModule())
def RreluWithNoiseBackwardTrainModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 5), tu.rand(3, 4, 5))


class RreluWithNoiseBackwardTrainStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5], torch.float32, True),
            ([3, 4, 5], torch.float32, True),
            ([3, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, grad, input, noise):
        return torch.ops.aten.rrelu_with_noise_backward(
            grad,
            input,
            noise,
            lower=0.1,
            upper=0.9,
            training=True,
            self_is_result=False,
        )


@register_test_case(module_factory=lambda: RreluWithNoiseBackwardTrainStaticModule())
def RreluWithNoiseBackwardTrainStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 5), tu.rand(3, 4, 5))


# ==============================================================================


class RreluWithNoiseBackwardEvalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, grad, input, noise):
        return torch.ops.aten.rrelu_with_noise_backward(
            grad,
            input,
            noise,
            lower=0.1,
            upper=0.9,
            training=False,
            self_is_result=False,
        )


@register_test_case(module_factory=lambda: RreluWithNoiseBackwardEvalModule())
def RreluWithNoiseBackwardEvalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 5), tu.rand(3, 4, 5))


class RreluWithNoiseBackwardEvalStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5], torch.float32, True),
            ([3, 4, 5], torch.float32, True),
            ([3, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, grad, input, noise):
        return torch.ops.aten.rrelu_with_noise_backward(
            grad,
            input,
            noise,
            lower=0.1,
            upper=0.9,
            training=False,
            self_is_result=False,
        )


@register_test_case(module_factory=lambda: RreluWithNoiseBackwardEvalStaticModule())
def RreluWithNoiseBackwardEvalStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 5), tu.rand(3, 4, 5))


class RreluWithNoiseForwardBackwardModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, grad, input, noise):
        res = torch.ops.aten.rrelu_with_noise_backward(
            grad,
            input,
            noise,
            lower=0.4,
            upper=0.6,
            training=True,
            self_is_result=False,
        )
        return torch.mean(res), torch.std(res)


@register_test_case(module_factory=lambda: RreluWithNoiseForwardBackwardModule())
def RreluWithNoiseForwardBackwardModule_basic(module, tu: TestUtils):
    grad = tu.rand(256, 244)
    input = tu.rand(256, 244, low=-1.0, high=1.0)
    noise = tu.rand(256, 244)
    torch.ops.aten.rrelu_with_noise(input, noise, lower=0.4, upper=0.6, training=True)
    module.forward(grad, input, noise)
