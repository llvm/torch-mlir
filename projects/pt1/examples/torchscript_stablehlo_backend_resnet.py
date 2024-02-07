import torch
import torchvision.models as models
from torch_mlir import torchscript

model = models.resnet18(pretrained=True)
model.eval()
data = torch.randn(2,3,200,200)
out_stablehlo_mlir_path = "./resnet18_stablehlo.mlir"

module = torchscript.compile(model, data, output_type=torchscript.OutputType.STABLEHLO, use_tracing=False)
with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))

print(f"StableHLO IR of resent18 successfully written into {out_stablehlo_mlir_path}")
