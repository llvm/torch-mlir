# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir import ir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from torch_mlir.extras.fx_importer import FxImporter


class SdpaModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.scaled_dot_product_attention.default(
            x,
            x,
            x,
            None,
            0.0,
            False,
            scale=None,
            enable_gqa=False,
        )


def lower_sdpa() -> None:
    module = SdpaModule().eval()

    example_input = torch.randn(2, 4, 8)
    exported = torch.export.export(module, (example_input,))

    decomposition_table = get_decomposition_table()
    if decomposition_table:
        exported = exported.run_decompositions(decomposition_table)

    context = ir.Context()
    torch_d.register_dialect(context)
    importer = FxImporter(context=context)
    importer.import_frozen_program(exported)
    mlir_module = importer.module

    pipeline = """
      builtin.module(
        func.func(torch-match-quantized-custom-ops),
        torchdynamo-export-to-torch-backend-pipeline{extra-library= backend-legal-ops=aten.as_strided},
        torch-adjust-calling-conventions
      )
    """
    run_pipeline_with_repro_report(
        mlir_module,
        pipeline,
        "Lowering TorchFX IR -> Torch Backend IR",
        enable_ir_printing=False,
    )

    module_str = str(mlir_module.operation)
    if "torch.aten.scaled_dot_product_attention.default" in module_str:
        raise RuntimeError(
            "scaled_dot_product_attention unexpectedly survived lowering"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    if hasattr(torch, "set_deterministic_debug_mode"):
        torch.set_deterministic_debug_mode("error")

    lower_sdpa()
    print("lowered scaled dot product attention")
    print("SUCCESS")
# CHECK: lowered scaled dot product attention
# CHECK: SUCCESS
