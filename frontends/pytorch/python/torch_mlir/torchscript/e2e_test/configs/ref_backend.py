#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

import numpy as np
import torch

import torch_mlir
from npcomp.compiler.pytorch.backend import refjit, frontend_lowering
from torch_mlir.torchscript.e2e_test.framework import TestConfig, Trace, TraceItem
from torch_mlir.torchscript.annotations import extract_annotations


class RefBackendTestConfig(TestConfig):
    """TestConfig that just runs the torch.nn.Module through RefBackend."""
    def __init__(self):
        super().__init__()
        self.backend = refjit.CompilerBackend()

    def compile(self, program: torch.nn.Module) -> Any:
        mb = torch_mlir.ModuleBuilder()
        scripted = torch.jit.script(program)
        class_annotator = torch_mlir.ClassAnnotator()

        extract_annotations(program, scripted, class_annotator)

        mb.import_module(scripted._c, class_annotator)
        # Lower module in place.
        frontend_lowering.lower_object_graph(mb.module)
        return self.backend.compile(mb.module)

    def run(self, artifact: Any, trace: Trace) -> Trace:
        jit_module = self.backend.load(artifact)
        result: Trace = []
        for item in trace:
            numpy_inputs = [t.numpy() for t in item.inputs]
            outputs = getattr(jit_module, item.symbol)(*numpy_inputs)
            if isinstance(outputs, np.ndarray):
                outputs = [outputs]
            torch_outputs = [torch.tensor(ndarray) for ndarray in outputs]
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          outputs=torch_outputs))
        return result
