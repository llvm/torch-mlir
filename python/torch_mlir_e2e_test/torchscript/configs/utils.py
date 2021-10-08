# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from typing import Any
from io import StringIO

import numpy as np
import torch

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report

def recursively_convert_to_numpy(o: Any):
    if isinstance(o, torch.Tensor):
        return o.numpy()
    if isinstance(o, tuple):
        return tuple(recursively_convert_to_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_to_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_to_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function input: {o}")

def recursively_convert_from_numpy(o: Any):
    if isinstance(o, np.ndarray):
        return torch.from_numpy(o)
    if isinstance(o, tuple):
        return tuple(recursively_convert_from_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_from_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_from_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function output: {o}")


def convert_torchscript_module_to_torch_backend_contract_mlir(program: torch.nn.Module):
    """Perform common lowering from TorchScript to Torch MLIR

    Returns an MLIR module that satisfies the Torch backend contract.
    """
    mb = ModuleBuilder()
    scripted = torch.jit.script(program)
    class_annotator = ClassAnnotator()

    extract_annotations(program, scripted, class_annotator)


    # TODO: Find a way to make each of these calls own its own
    # "debuggable error report" situation.
    try:
        original_stderr = sys.stderr
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
        sys.stderr = original_stderr

    run_pipeline_with_repro_report(
        mb.module,
        "torchscript-module-to-torch-backend-pipeline",
        "Lowering TorchScript Object Graph IR -> Torch Backend IR")

    return mb.module
