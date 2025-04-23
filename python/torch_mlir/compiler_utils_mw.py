# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
from enum import Enum
from io import StringIO
import os
import sys
import tempfile
from typing import Union, List

import torch
from .passmanager import PassManager
from .ir import StringAttr

from torch_mlir.compiler_utils import OutputType


def run_pipeline_mw(
    module, pipeline: str, description: str, enable_ir_printing: bool = False
):
    """Runs `pipeline` on `module`"""
    with module.context as ctx:
        # TODO(#3506): Passes can emit errors but not signal failure,
        # which causes a native assert.
        ctx.emit_error_diagnostics = False
        pm = PassManager.parse(pipeline)
        if enable_ir_printing:
            ctx.enable_multithreading(False)
            pm.enable_ir_printing()
        pm.run(module.operation)


def lower_mlir_module_mw(verbose, output_type, module):
    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(module)

    if output_type == OutputType.TORCH:
        return module

    if output_type == OutputType.TOSA:
        run_pipeline_mw(
            module,
            "builtin.module(torch-backend-to-tosa-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA Backend IR",
        )
        if verbose:
            print("\n====================")
            print("TOSA Backend IR")
            print(module)
        return module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_mw(
            module,
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
        )
        if verbose:
            print("\n====================")
            print("LINALG Backend IR")
            print(module)
        return module

    elif output_type == OutputType.TOSA_LINALG:
        run_pipeline_mw(
            module,
            "builtin.module(torch-backend-to-tosa-linalg-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA_LINALG Backend IR",
        )
        if verbose:
            print("\n====================")
            print("TODA_LINALG Backend IR")
            print(module)
        return module
    raise Exception(f"Unknown OutputType: {output_type}")
