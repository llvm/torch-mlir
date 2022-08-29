# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch.utils._pytree import tree_map

from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor
from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem


def wrap(e):
    return TorchMLIRTensor(e.detach().clone()) if isinstance(e, torch.Tensor) else e


def unwrap(e):
    return TorchMLIRTensor.unwrap(e) if isinstance(e, TorchMLIRTensor) else e


def to_tmt(m: torch.nn.Module):
    for buf_name, buf in m.named_buffers(recurse=True):
        if isinstance(buf, TorchMLIRTensor):
            continue
        m.register_buffer(buf_name, TorchMLIRTensor(buf))
    for param_name, param in m.named_parameters(recurse=True):
        if isinstance(param, TorchMLIRTensor):
            continue
        m.register_parameter(
            param_name,
            torch.nn.Parameter(
                TorchMLIRTensor(param), requires_grad=param.requires_grad
            ),
        )
    for attr in dir(m):
        field = getattr(m, attr)
        if isinstance(field, torch.Tensor) and not isinstance(field, TorchMLIRTensor):
            setattr(m, attr, TorchMLIRTensor(field))


class EagerModeTestConfig(TestConfig):
    """Trivial test config that exercises eager mode plumbing"""

    def __init__(self):
        super().__init__()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        program.apply(to_tmt)
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []
        for item in trace:
            attr = artifact
            for part in item.symbol.split("."):
                attr = getattr(attr, part)

            inps = tree_map(wrap, item.inputs)
            outps = attr(*inps)
            output = tree_map(unwrap, outps)

            result.append(
                TraceItem(symbol=item.symbol, inputs=item.inputs, output=output)
            )
        return result
