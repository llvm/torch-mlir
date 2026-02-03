# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Optional, Union, Dict, Tuple, Any, Callable
from packaging import version
from dataclasses import dataclass

import warnings

import torch
import torch.export
import torch.nn as nn
from torch.export import ExportedProgram

from .extras.fx_importer import FxImporter, FxImporterHooks
from . import ir
from .dialects import torch as torch_d
from .extras.fx_decomp_util import get_decomposition_table
from .compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
    lower_mlir_module,
    BackendLoweringOptions,
)


@dataclass
class FxImportOptions:
    """Options for FX graph import and lowering."""

    extra_library_file_name: Optional[str] = None
    backend_legal_ops: Optional[list[str]] = None


def _module_lowering(
    verbose,
    enable_ir_printing,
    output_type,
    torch_mod,
    fx_import_options: Optional[FxImportOptions] = None,
    backend_options: Optional[BackendLoweringOptions] = None,
):

    if fx_import_options is None:
        fx_import_options = FxImportOptions()

    if backend_options is None:
        backend_options = BackendLoweringOptions()

    if verbose:
        print("\n====================")
        print("TorchFX IR")
        print(torch_mod)

    if output_type == OutputType.RAW:
        return torch_mod
    # TODO: pass extra_library_file_name by caller

    backend_legal_op_arg_str = ""
    if fx_import_options.backend_legal_ops is not None:
        if not len(fx_import_options.backend_legal_ops) == 0:
            backend_legal_op_arg_str = "backend-legal-ops=" + ",".join(
                fx_import_options.backend_legal_ops
            )

    if fx_import_options.extra_library_file_name is None:
        extra_library_file_name = ""
    else:
        extra_library_file_name = fx_import_options.extra_library_file_name
    option_string = (
        "{"
        + backend_legal_op_arg_str
        + " extra-library="
        + extra_library_file_name
        + "}"
    )

    run_pipeline_with_repro_report(
        torch_mod,
        f"builtin.module(func.func(torch-match-quantized-custom-ops), torchdynamo-export-to-torch-backend-pipeline{option_string})",
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=enable_ir_printing,
    )
    return lower_mlir_module(verbose, output_type, torch_mod, backend_options)


def export_and_import(
    f: Union[nn.Module, ExportedProgram],
    *args,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    strict: bool = False,
    experimental_support_mutation: bool = False,
    import_symbolic_shape_expressions: bool = False,
    hooks: Optional[FxImporterHooks] = None,
    decomposition_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None,
    func_name: str = "main",
    enable_graph_printing: bool = False,
    verbose: bool = False,
    enable_ir_printing: bool = False,
    backend_legal_ops: Optional[list[str]] = None,
    allow_non_finites: bool = True,
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
            prog = torch.export.export(
                f, args, kwargs, dynamic_shapes=dynamic_shapes, strict=strict
            )
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

    fx_import_options = FxImportOptions(backend_legal_ops=backend_legal_ops)
    backend_options = BackendLoweringOptions(allow_non_finites=allow_non_finites)

    return _module_lowering(
        verbose,
        enable_ir_printing,
        OutputType.get(output_type),
        fx_importer.module,
        fx_import_options=fx_import_options,
        backend_options=backend_options,
    )


def stateless_fx_import(
    gm: torch.fx.GraphModule,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    hooks: Optional[FxImporterHooks] = None,
    model_name: str = "main",
    enable_graph_printing: bool = False,
    verbose: bool = False,
    enable_ir_printing: bool = False,
    backend_legal_ops: Optional[list[str]] = None,
    allow_non_finites: bool = True,
):
    if enable_graph_printing:
        gm.print_readable()
    context = ir.Context()
    torch_d.register_dialect(context)
    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)
    fx_importer.import_stateless_graph(gm.graph, func_name=model_name)

    fx_import_options = FxImportOptions(backend_legal_ops=backend_legal_ops)
    backend_options = BackendLoweringOptions(allow_non_finites=allow_non_finites)

    return _module_lowering(
        verbose,
        enable_ir_printing,
        OutputType.get(output_type),
        fx_importer.module,
        fx_import_options=fx_import_options,
        backend_options=backend_options,
    )
