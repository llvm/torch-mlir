# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

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
    module.forward(torch.randn(3, 2, 4), torch.randn(3, 2, 4))


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
    module.forward(torch.randn(3, 3), torch.randn(3, 3))


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
    module.forward(torch.randn(3, 2, 4), torch.randn(3, 2, 4))
