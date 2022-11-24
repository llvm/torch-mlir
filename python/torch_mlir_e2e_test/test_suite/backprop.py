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
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, grad_output, output):
        return torch.ops.aten._softmax_backward_data(grad_output,
                                                     output,
                                                     dim=1,
                                                     input_dtype=6)


@register_test_case(module_factory=lambda: SoftmaxBackwardModule())
def SoftmaxBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))


# ==============================================================================
class TanhBackwardModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, grad_out, output):
        return torch.ops.aten.tanh_backward(grad_out, output)


@register_test_case(module_factory=lambda: TanhBackwardModule())
def TanhBackward_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3), tu.rand(3, 3))


# ==============================================================================

class ConvolutionBackwardModule2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
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
            output_mask=[True, True, True])


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2D())
def ConvolutionBackwardModule2D_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(tu.rand(2, 2, 5, 5), tu.rand(2, 2, 6, 6),
                       tu.rand(2, 2, 2, 2))


class ConvolutionBackwardModule2DPadded(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
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
            output_mask=[True, True, True])


@register_test_case(module_factory=lambda: ConvolutionBackwardModule2DPadded())
def ConvolutionBackwardModule2DPadded_basic(module, tu: TestUtils):
    with torch.backends.mkldnn.flags(enabled=False):
        module.forward(tu.rand(2, 2, 8, 8), tu.rand(2, 2, 6, 6),
                       tu.rand(2, 2, 3, 3))


# ==============================================================================


class GeluBackwardModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, grad, input):
        return torch.ops.aten.gelu_backward(grad, input)


@register_test_case(module_factory=lambda: GeluBackwardModule())
def GeluBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3), tu.rand(5, 3))


class LogSoftmaxBackwardModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, grad_output, output):
        return torch.ops.aten._log_softmax_backward_data(grad_output,
                                                         output,
                                                         dim=1,
                                                         input_dtype=6)


@register_test_case(module_factory=lambda: LogSoftmaxBackwardModule())
def LogSoftmaxBackwardModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))
