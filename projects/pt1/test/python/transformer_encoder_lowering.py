# RUN: %PYTHON %s | FileCheck %s

import torch
from torch.nn import TransformerEncoderLayer

from torch_mlir import ir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from torch_mlir.extras.fx_importer import FxImporter


def lower_transformer(norm_first: bool, activation: str) -> None:
    layer = TransformerEncoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        activation=activation,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=norm_first,
        bias=True,
    ).eval()

    example_input = torch.randn(2, 5, 32)
    exported = torch.export.export(layer, (example_input,))

    decomposition_table = get_decomposition_table()
    if decomposition_table:
        exported = exported.run_decompositions(decomposition_table)

    context = ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    importer.import_frozen_program(exported)
    module = importer.module

    pipeline = """
      builtin.module(
        func.func(torch-match-quantized-custom-ops),
        torchdynamo-export-to-torch-backend-pipeline{extra-library= backend-legal-ops=aten.as_strided},
        torch-adjust-calling-conventions
      )
    """
    run_pipeline_with_repro_report(
        module,
        pipeline,
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )

    if "torch.operator" in str(module.operation):
        raise RuntimeError("Unexpected torch.operator after lowering")


if __name__ == "__main__":
    torch.manual_seed(0)
    if hasattr(torch, "set_deterministic_debug_mode"):
        torch.set_deterministic_debug_mode("error")

    for activation in ("gelu", "relu"):
        for norm_first in (True, False):
            lower_transformer(norm_first, activation)
            print(
                f"CHECK: lowered norm_first={norm_first} activation={activation}"
            )

    print("SUCCESS")
# CHECK: CHECK: lowered norm_first=True activation=gelu
# CHECK: CHECK: lowered norm_first=False activation=gelu
# CHECK: CHECK: lowered norm_first=True activation=relu
# CHECK: CHECK: lowered norm_first=False activation=relu
# CHECK: SUCCESS
