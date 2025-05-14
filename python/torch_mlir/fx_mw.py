# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch_mlir
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

    mlir_module = fx.export_and_import(
        prog,
        output_type=OutputType.RAW,
        experimental_support_mutation=experimental_support_mutation,
    )

    if output_type != "raw":
        mlir_module = lower_module(mlir_module, output_type)

    return mlir_module


def lower_module_from_file(mlir_file: str, output_type: str):
    src = open(mlir_file, "r").read()
    with torch_mlir.ir.Context() as ctx:
        torch_mlir.dialects.torch.register_dialect(ctx)
        with torch_mlir.ir.Location.unknown() as loc:
            mlir_module = torch_mlir.ir.Module.parse(src)

    return lower_module(mlir_module, output_type)


def lower_module(mlir_module, output_type: str):

    backend_legal_ops = None

    match output_type:
        case "torch":
            output_type = OutputType.TORCH
        case "tosa":
            output_type = OutputType.TOSA
            backend_legal_ops = [
                "aten.flatten.using_ints",
                "aten.native_layer_norm",
                "aten.adaptive_avg_pool1d",
                "aten.adaptive_avg_pool2d",
                "aten.adaptive_max_pool1d",
                "aten.adaptive_max_pool2d",
                "aten.linear",
            ]
        case "linalg_on_tensors":
            output_type = OutputType.LINALG_ON_TENSORS
            backend_legal_ops = [
                "aten.flatten.using_ints",
                "aten.adaptive_avg_pool1d",
                "aten.adaptive_avg_pool2d",
                "aten.adaptive_max_pool1d",
                "aten.adaptive_max_pool2d",
                "aten.unflatten.int",
            ]
        case "tosa_linalg":
            output_type = OutputType.TOSA_LINALG
            backend_legal_ops = [
                "aten.flatten.using_ints",
                "aten.native_layer_norm",
                "aten.adaptive_avg_pool1d",
                "aten.adaptive_avg_pool2d",
                "aten.adaptive_max_pool1d",
                "aten.adaptive_max_pool2d",
                "aten.linear",
                "aten.unflatten.int",
            ]
        case "raw":
            output_type = OutputType.RAW
        case _:
            raise ValueError("Importing PyTorch model failed: Unsupported output type.")

    backend_legal_op_arg_str = ""
    if backend_legal_ops is not None:
        if not len(backend_legal_ops) == 0:
            backend_legal_op_arg_str = "backend-legal-ops=" + ",".join(
                backend_legal_ops
            )

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
    return lower_mlir_module_mw(verbose, output_type, mlir_module)
