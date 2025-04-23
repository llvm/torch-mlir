# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from .compiler_utils import OutputType

from .compiler_utils_mw import (
    run_pipeline_mw,
    lower_mlir_module_mw,
)

from . import fx
from torch._decomp import get_decompositions


def import_exported_model(
    prog: torch.export.ExportedProgram,
    output_type: str,
    experimental_support_mutation: bool = True,
):

    decomp_table = get_decompositions(
        [torch.ops.aten.lstm.input, torch.ops.aten.gru.input]
    )

    prog = prog.run_decompositions(decomp_table)

    match output_type:
        case "torch":
            output_type = OutputType.TORCH
        case "tosa":
            output_type = OutputType.TOSA
        case "linalg_on_tensors":
            output_type = OutputType.LINALG_ON_TENSORS
        case "tosa_linalg":
            output_type = OutputType.TOSA_LINALG
        case "raw":
            output_type = OutputType.RAW
        case _:
            raise ValueError("Importing PyTorch model failed: Unsupported output type.")

    mlir_module = fx.export_and_import(
        prog,
        output_type=OutputType.RAW,
        experimental_support_mutation=experimental_support_mutation,
    )

    if output_type != OutputType.RAW:
        backend_legal_op_arg_str = ""
        extra_library_file_name = ""
        option_string = (
            "{"
            + backend_legal_op_arg_str
            + " extra-library="
            + extra_library_file_name
            + "}"
        )
        run_pipeline_mw(
            mlir_module,
            f"builtin.module(func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline{option_string})",
            "Lowering TorchFX IR -> Torch Backend IR",
            enable_ir_printing=False,
        )
        verbose = False
        mlir_module = lower_mlir_module_mw(verbose, output_type, mlir_module)

    return mlir_module
