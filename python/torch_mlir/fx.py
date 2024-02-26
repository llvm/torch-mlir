# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Optional

import warnings

import torch
import torch.export
import torch.nn as nn

from torch_mlir.extras.fx_importer import FxImporter, FxImporterHooks
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_decomp_util import get_decomposition_table

def export_and_import(
    f,
    *args,
    fx_importer: Optional[FxImporter] = None,
    constraints: Optional[torch.export.Constraint] = None,
    experimental_support_mutation: bool = False,
    hooks: Optional[FxImporterHooks] = None,
    func_name: str = "main",
    **kwargs,
):
    context = ir.Context()
    torch_d.register_dialect(context)

    if fx_importer is None:
        fx_importer = FxImporter(context=context, hooks=hooks)
    prog = torch.export.export(f, args, kwargs, constraints=constraints)
    decomp_table = get_decomposition_table()
    prog = prog.run_decompositions(decomp_table)
    if experimental_support_mutation:
        if torch.__version__ < "2.3.0.dev20240207":
            warnings.warn("Mutable program import only supported on PyTorch 2.3+")
        fx_importer.import_program(prog, func_name=func_name)
    else:
        fx_importer.import_frozen_program(prog, func_name=func_name)

    return fx_importer.module_op
