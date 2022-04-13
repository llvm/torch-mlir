# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch.utils._pytree import tree_map

from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor
from torch_mlir_e2e_test.torchscript.framework import TestConfig, Trace, TraceItem


def wrap(e):
    return TorchMLIRTensor(e.detach().clone()) if isinstance(e, torch.Tensor) else e


def unwrap(e):
    return TorchMLIRTensor.unwrap(e) if isinstance(e, TorchMLIRTensor) else e


class EagerModeTestConfig(TestConfig):
    """Trivial test config that exercises eager mode plumbing"""

    def __init__(self):
        super().__init__()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []
        for item in trace:
            attr = artifact
            for part in item.symbol.split('.'):
                attr = getattr(attr, part)

            inps = tree_map(wrap, item.inputs)
            outps = attr(*inps)
            output = tree_map(unwrap, outps)

            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
