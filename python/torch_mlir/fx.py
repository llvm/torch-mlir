from typing import Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.decomp_util import get_decomposition_table

# model_name is for users that are running a model through the fx importer.
# Based on the model, there may be some export/import optimizations made.
# Currently, it is used to deal with OPT-125M (model_name = "opt-125M") that generates extra args
# when exporting through fx. (optional arg, don't have to pass anything)
def export_and_import(
    f,
    *args,
    fx_importer: Optional[FxImporter] = None,
    constraints: Optional[torch.export.Constraint] = None,
    model_name: Optional[str] = None,
    **kwargs,
):
    context = ir.Context()
    torch_d.register_dialect(context)

    if fx_importer is None:
        fx_importer = FxImporter(context=context)
    prog = torch.export.export(f, args, kwargs, constraints=constraints)
    decomp_table = get_decomposition_table(model_name)
    prog = prog.run_decompositions(decomp_table)
    fx_importer.import_frozen_exported_program(prog)
    return fx_importer.module_op
