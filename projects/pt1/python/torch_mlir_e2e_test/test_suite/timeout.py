# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export


# ==============================================================================
class TimeOutModule(torch.nn.Module):
    """
    This test ensures that the timeout mechanism works as expected.

    The module runs an infinite loop that will never terminate,
    and the test is expected to time out and get terminated
    """

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, x):
        """
        Run an infinite loop.

        This may loop in the compiler or the runtime depending on whether
        fx or torchscript is used.
        """
        # input_arg_2 is going to be 2
        # but we can't just specify it as a
        # constant because the compiler will
        # attempt to get rid of the whole loop
        input_arg_2 = x.size(0)
        sum = 100
        while input_arg_2 < sum:  # sum will always > 2
            sum += 1
        return sum


@register_test_case(module_factory=lambda: TimeOutModule(), timeout_seconds=10)
def TimeOutModule_basic(module, tu: TestUtils):
    module.forward(torch.ones((42, 42)))
