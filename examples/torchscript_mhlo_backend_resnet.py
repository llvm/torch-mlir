import torch
import torchvision.models as models
import torch_mlir

model = models.resnet18(pretrained=True)
model.eval()
data = torch.randn(2,3,200,200)
out_mhlo_mlir_path = "./resnet18_mhlo.mlir"

module = torch_mlir.compile(model, data, output_type=torch_mlir.OutputType.MHLO, use_tracing=False)
with open(out_mhlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))

print(f"MHLO IR of resent18 successfully written into {out_mhlo_mlir_path}")
