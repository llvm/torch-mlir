import torch
import torchvision.models as models
import torch_mlir

model = models.resnet18(pretrained=True)
model.eval()
data = torch.randn(2,3,200,200)
out_stablehlo_mlir_path = "./resnet18_stablehlo.mlir"

module = torch_mlir.compile(model, data, output_type=torch_mlir.OutputType.STABLEHLO, use_tracing=False)
with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))

print(f"StableHLO IR of resent18 successfully written into {out_stablehlo_mlir_path}")
