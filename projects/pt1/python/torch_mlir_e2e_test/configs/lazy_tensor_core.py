# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend
import torch
from torch.utils._pytree import tree_map

from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem


def to_device(device):
    """Returns a lambda that maps `torch.Tensor` objects to `device`, and ignores other types"""
    return lambda e: e.to(device) if isinstance(e, torch.Tensor) else e


class LazyTensorCoreTestConfig(TestConfig):
    """TestConfig that runs torch.nn.Module thru the Lazy Tensor Core frontend for Torch MLIR"""

    def __init__(self):
        super().__init__()
        lazy_backend._initialize()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program.to('lazy')

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []

        for item in trace:
            # We need to move all the inputs to the lazy device before running in LTC.
            lazy_inputs = tree_map(to_device('lazy'), item.inputs)
            output = getattr(artifact, item.symbol)(*lazy_inputs)
            cpu_outputs = tree_map(to_device('cpu'), output)

            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=cpu_outputs))

        return result
