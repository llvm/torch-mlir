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

import torch_mlir
import npcomp
from npcomp.passmanager import PassManager
from npcomp.compiler.pytorch.backend import refjit
from npcomp.compiler.pytorch.backend.abc import NpcompBackend
from npcomp_torchscript.e2e_test.framework import TestConfig, Trace, TraceItem
from torch_mlir.torchscript_annotations import extract_annotations

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

class NpcompBackendTestConfig(TestConfig):
    """Base class for TestConfig's that are implemented with npcomp.

    This class handles all the common lowering that npcomp does before reaching
    its backends.
    """
    def __init__(self, backend: NpcompBackend):
        super().__init__()
        self.backend = backend

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

        # The torch-mlir python code is built against its own aggregate CAPI.
        # The npcomp python module is built against our own.
        # So we need to transport it across those as a string.
        with npcomp.ir.Context() as ctx:
            npcomp.register_all_dialects(ctx)
            module = npcomp.ir.Module.parse(str(mb.module))

        try:
            sys.stderr = StringIO()
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10, enable_debug_info=True)
            pipeline_str = "torchscript-to-npcomp-backend-pipeline"
            # Lower module in place to make it ready for compiler backends.
            with module.context:
                pm = PassManager.parse(pipeline_str)
                pm.run(module)
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

        try:
            sys.stderr = StringIO()
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10, enable_debug_info=True)
            return self.backend.compile(module)
        except Exception as e:
            filename = os.path.join(tempfile.gettempdir(),
                                    scripted.original_name + '.mlir')
            with open(filename, 'w') as f:
                f.write(asm_for_error_report)
            raise Exception(f"""
NPCOMP Backend lowering for {self.backend.__class__.__name__} failed with the following diagnostics:
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
