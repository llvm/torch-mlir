# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any

import torch
from torch_mlir import torchscript

from torch_mlir_e2e_test.tosa_backends.abc import TosaBackend
from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem
from torch_mlir_e2e_test.utils import convert_annotations_to_placeholders
from .utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)


class TosaBackendTestConfig(TestConfig):
    """Base class for TestConfig's that are implemented with TOSA.

    This class handles all the common lowering that torch-mlir does before
    reaching the TOSA abstraction level.
    """
    def __init__(self, backend: TosaBackend, use_make_fx: bool = False):
        super().__init__()
        self.backend = backend
        self.use_make_fx = use_make_fx

    def compile(self, program: torch.nn.Module) -> Any:
        example_args = convert_annotations_to_placeholders(program.forward)
        module = torchscript.compile(
            program, example_args, output_type="tosa", use_make_fx=self.use_make_fx)

        return self.backend.compile(module)



    def run(self, artifact: Any, trace: Trace) -> Trace:
        backend_module = self.backend.load(artifact)
        result: Trace = []
        for item in trace:
            numpy_inputs = recursively_convert_to_numpy(item.inputs)
            outputs = getattr(backend_module, item.symbol)(*numpy_inputs)
            output = recursively_convert_from_numpy(outputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
