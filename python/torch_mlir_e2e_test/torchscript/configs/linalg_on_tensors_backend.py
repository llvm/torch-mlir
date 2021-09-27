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

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.framework import TestConfig, Trace, TraceItem

def _recursively_convert_to_numpy(o: Any):
    if isinstance(o, torch.Tensor):
        return o.numpy()
    if isinstance(o, tuple):
        return tuple(_recursively_convert_to_numpy(x) for x in o)
    if isinstance(o, list):
        return [_recursively_convert_to_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: _recursively_convert_to_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function input: {o}")

def _recursively_convert_from_numpy(o: Any):
    if isinstance(o, np.ndarray):
        return torch.from_numpy(o)
    if isinstance(o, tuple):
        return tuple(_recursively_convert_from_numpy(x) for x in o)
    if isinstance(o, list):
        return [_recursively_convert_from_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: _recursively_convert_from_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function output: {o}")

class LinalgOnTensorsBackendTestConfig(TestConfig):
    """Base class for TestConfig's that are implemented with linalg-on-tensors.

    This class handles all the common lowering that torch-mlir does before
    reaching the linalg-on-tensors abstraction level.
    """
    def __init__(self, backend: LinalgOnTensorsBackend):
        super().__init__()
        self.backend = backend

    def compile(self, program: torch.nn.Module) -> Any:
        mb = ModuleBuilder()
        scripted = torch.jit.script(program)
        class_annotator = ClassAnnotator()

        extract_annotations(program, scripted, class_annotator)

        # TODO: Find a way to make each of these calls own its own
        # "debuggable error report" situation.
        try:
            sys.stderr = StringIO()
            # Import the TorchScript module to MLIR
            mb.import_module(scripted._c, class_annotator)
        except Exception as e:
            raise Exception(f"""
PyTorch TorchScript module -> torch-mlir Object Graph IR import failed with:
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
            pipeline_str = "torchscript-module-to-linalg-on-tensors-backend-pipeline"
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
torch-mlir TorchScript Object Graph IR -> linalg-on-tensors backend IR lowering failed with the following diagnostics:
{sys.stderr.getvalue()}

Error can be reproduced with:
$ torch-mlir-opt -{pipeline_str} {filename}
""") from None
        finally:
            sys.stderr = sys.__stderr__

        try:
            sys.stderr = StringIO()
            asm_for_error_report = mb.module.operation.get_asm(
                large_elements_limit=10, enable_debug_info=True)
            return self.backend.compile(mb.module)
        except Exception as e:
            filename = os.path.join(tempfile.gettempdir(),
                                    scripted.original_name + '.mlir')
            with open(filename, 'w') as f:
                f.write(asm_for_error_report)
            raise Exception(f"""
torch-mlir linalg-on-tensors Backend lowering for {self.backend.__class__.__name__} failed with the following diagnostics:
## Exception:
{e}

## Stderr:
{sys.stderr.getvalue()}

## Input IR has been saved in {filename}
""") from None
        finally:
            sys.stderr = sys.__stderr__


    def run(self, artifact: Any, trace: Trace) -> Trace:
        backend_module = self.backend.load(artifact)
        result: Trace = []
        for item in trace:
            numpy_inputs = _recursively_convert_to_numpy(item.inputs)
            outputs = getattr(backend_module, item.symbol)(*numpy_inputs)
            output = _recursively_convert_from_numpy(outputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
