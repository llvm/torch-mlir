import torch
import torch_mlir

from transformers import BertForMaskedLM

# Wrap the bert model to avoid multiple returns problem
class BertTinyWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny", return_dict=False)
    
    def forward(self, data):
        return self.bert(data)[0]

model = BertTinyWrapper()
model.eval()
data = torch.randint(30522, (2, 128))
out_stablehlo_mlir_path = "./bert_tiny_stablehlo.mlir"

module = torch_mlir.compile(model, data, output_type=torch_mlir.OutputType.STABLEHLO, use_tracing=True)
with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))

print(f"StableHLO IR of tiny bert successfully written into {out_stablehlo_mlir_path}")
