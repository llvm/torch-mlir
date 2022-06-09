# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import ltc_backend.ltc_backend._EXAMPLE_MLIR_BACKEND as ltc_backend
import torch
from torch_mlir_e2e_test.torchscript.framework import TestConfig, Trace, TraceItem


class LazyTensorCoreTestConfig(TestConfig):
    """TestConfig that runs torch.nn.Module thru the Lazy Tensor Core frontend for Torch MLIR"""

    def __init__(self):
        super().__init__()
        ltc_backend._initialize()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program.to('lazy')

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []

        for item in trace:
            # We need to move all the inputs to the lazy device before running in LTC.
            lazy_inputs = [arg.to('lazy') for arg in item.inputs]
            output = getattr(artifact, item.symbol)(*lazy_inputs)

            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output.to('cpu')))

        return result
