# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import copy
from typing import Any

import torch

from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem


class TorchScriptTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module through TorchScript"""
    def __init__(self):
        super().__init__()

    def compile(self, program: torch.nn.Module) -> torch.jit.ScriptModule:
        return torch.jit.script(program)

    def run(self, artifact: torch.jit.ScriptModule, trace: Trace) -> Trace:
        # TODO: Deepcopy the torch.jit.ScriptModule, so that if the program is
        # stateful then it does not mutate the original compiled program.

        result: Trace = []
        for item in trace:
            attr = artifact
            for part in item.symbol.split('.'):
                attr = getattr(attr, part)
            output = attr(*item.inputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
