#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from typing import Any
from io import StringIO
import os
import tempfile

import numpy as np
import torch
from mlir.passmanager import PassManager

import torch_mlir
from npcomp.compiler.pytorch.backend import refjit
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

        # TODO: Find a way to make each of these calls own its own
        # "debuggable error report" situation.
        try:
            sys.stderr = StringIO()
            # Import the TorchScript module to MLIR
            mb.import_module(scripted._c, class_annotator)
        except Exception as e:
            raise Exception(f"""
PyTorch TorchScript module -> NPCOMP Object Graph IR import failed with:
Exception:
{e}
Diagnostics:
{sys.stderr.getvalue()}
""") from None
        finally:
            sys.stderr = sys.__stderr__

        try:
            sys.stderr = StringIO()
            asm_for_error_report = mb.module.operation.get_asm(
                large_elements_limit=10, enable_debug_info=True)
            pipeline_str = "torchscript-to-npcomp-backend-pipeline"
            # Lower module in place to make it ready for compiler backends.
            with mb.module.context:
                pm = PassManager.parse(pipeline_str)
                pm.run(mb.module)
        except Exception as e:
            # TODO: More robust.
            # - don't arbitrarily clutter up /tmp. When a test suite has many
            #   tests, this can be a big disk cost (also, /tmp/ is frequently a
            #   RAM fs, which increases worries about capacity).
            # - don't have colliding filenames (hard to do without cluttering
            #   up /tmp)
            # - if we do have have colliding filenames, writes should at least
            #   avoid being racy.
            filename = os.path.join(tempfile.gettempdir(),
                                    scripted.original_name + '.mlir')
            with open(filename, 'w') as f:
                f.write(asm_for_error_report)
            raise Exception(f"""
NPCOMP TorchScript Object Graph IR -> NPCOMP Backend IR lowering failed with the following diagnostics:
{sys.stderr.getvalue()}

Error can be reproduced with:
$ npcomp-opt -{pipeline_str} {filename}
""") from None
        finally:
            sys.stderr = sys.__stderr__
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
