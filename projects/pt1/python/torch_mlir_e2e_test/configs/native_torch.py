# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem


class NativeTorchTestConfig(TestConfig):
    """TestConfig that just runs the torch.nn.Module without compiling"""
    def __init__(self):
        super().__init__()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        # TODO: Deepcopy the torch.nn.Module, so that if the program is
        # stateful then it does not mutate the original compiled program.
        result: Trace = []
        for item in trace:
            output = getattr(artifact, item.symbol)(*item.inputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
