# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, Optional, Union, Dict, Tuple, Any, Callable
from packaging import version

import warnings

import torch
import torch.export
import torch.nn as nn
from torch.export import ExportedProgram
from torch.fx.experimental.symbolic_shapes import ShapeEnv

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


def _get_shape_env_from_inputs(inputs: List[torch.Tensor]) -> Optional[ShapeEnv]:
    """
    Finds the ShapeEnv object from the inputs. Returns None if it could not be
    found. This method is adapted from the module torch._inductor.compile_fx.
    """
    fake_mode = torch._dynamo.utils.detect_fake_mode(inputs)
    if fake_mode is not None:
        return fake_mode.shape_env
    for input in inputs:
        if isinstance(input, torch.SymInt):
            return input.node.shape_env
    return None


def _get_range_constraints(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    """
    Given a graph module and list of example inputs, this function returns the
    set of range constraints representing bounds on the sizes of input tensors.
    """
    shape_env = _get_shape_env_from_inputs(example_inputs)
    assert shape_env is not None
    range_constraints = {k: v for k, v in shape_env.var_to_range.items()}
    return range_constraints


def stateless_fx_import(
    gm: torch.fx.GraphModule,
    output_type: Union[str, OutputType] = OutputType.RAW,
    fx_importer: Optional[FxImporter] = None,
    hooks: Optional[FxImporterHooks] = None,
    model_name: str = "main",
    enable_graph_printing: bool = False,
    enable_ir_printing: bool = False,
    import_symbolic_shape_expressions: bool = False,
    example_inputs: List[Any] = None,
):
    if enable_graph_printing:
        gm.print_readable()
    context = ir.Context()
    torch_d.register_dialect(context)
    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)

    # Graph module does not contain the range constraints. We compute the
    # constraints here using the shape environment that is associated with
    # example inputs.
    assert (
        not import_symbolic_shape_expressions or example_inputs is not None
    ), "importing symbolic shape expressions requires example args to be provided"

    gm.meta["range_constraints"] = _get_range_constraints(gm, example_inputs)

    fx_importer.import_stateless_graph(
        gm.graph,
        func_name=model_name,
        import_symbolic_shape_expressions=import_symbolic_shape_expressions,
    )
    return _module_lowering(
        enable_ir_printing, OutputType.get(output_type), fx_importer.module
    )
