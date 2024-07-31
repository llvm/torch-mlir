# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Optional, Union, Dict, Tuple, Any, Callable
from packaging import version

import warnings

import torch
import torch.export
import torch.nn as nn
from torch.export import ExportedProgram

from torch_mlir.extras.fx_importer import FxImporter, FxImporterHooks
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
)


def _module_lowering(
    verbose,
    output_type,
    torch_mod,
    extra_library_file_name=None,
):

    if output_type == OutputType.RAW:
        if verbose:
            print(torch_mod)
        return torch_mod
    # TODO: pass extra_library_file_name by caller
    if extra_library_file_name is None:
        extra_library_file_name = ""
    option_string = "{extra-library=" + extra_library_file_name + "}"
    run_pipeline_with_repro_report(
        torch_mod,
        f"builtin.module(torchdynamo-export-to-torch-backend-pipeline{option_string})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=verbose,
    )
    return lower_mlir_module(verbose, output_type, torch_mod)


def export_and_import(
    f: Union[nn.Module, ExportedProgram],
    *args,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    experimental_support_mutation: bool = False,
    import_symbolic_shape_expressions: bool = False,
    hooks: Optional[FxImporterHooks] = None,
    decomposition_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None,
    func_name: str = "main",
    enable_graph_printing: bool = False,
    enable_ir_printing: bool = False,
    **kwargs,
):
    context = ir.Context()
    torch_d.register_dialect(context)

    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)
    if isinstance(f, ExportedProgram):
        prog = f
    else:
        # pytorch 2.1 or lower doesn't have `dyanmic_shapes` keyword argument in torch.export
        if version.Version(torch.__version__) >= version.Version("2.2.0"):
            prog = torch.export.export(f, args, kwargs, dynamic_shapes=dynamic_shapes)
        else:
            prog = torch.export.export(f, args, kwargs)
    if decomposition_table is None:
        decomposition_table = get_decomposition_table()
    if decomposition_table:
        prog = prog.run_decompositions(decomposition_table)
    if enable_graph_printing:
        prog.graph_module.print_readable()
    if experimental_support_mutation:
        if torch.__version__ < "2.3.0.dev20240207":
            warnings.warn("Mutable program import only supported on PyTorch 2.3+")
        fx_importer.import_program(
            prog,
            func_name=func_name,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )
    else:
        fx_importer.import_frozen_program(
            prog,
            func_name=func_name,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )

    return _module_lowering(
        enable_ir_printing, OutputType.get(output_type), fx_importer.module
    )


def stateless_fx_import(
    gm: torch.fx.GraphModule,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    hooks: Optional[FxImporterHooks] = None,
    model_name: str = "main",
    enable_graph_printing: bool = False,
    enable_ir_printing: bool = False,
):
    if enable_graph_printing:
        gm.print_readable()
    context = ir.Context()
    torch_d.register_dialect(context)
    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)
    fx_importer.import_stateless_graph(gm.graph, func_name=model_name)
    return _module_lowering(
        enable_ir_printing, OutputType.get(output_type), fx_importer.module
    )
