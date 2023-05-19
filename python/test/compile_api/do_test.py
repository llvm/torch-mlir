# RUN: %PYTHON %s

import torch_mlir
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return 2 * x

class ModelWithTuple(torch.nn.Module):
    def forward(self, x):
        return (2 * x,)
    
class ModelWithNestedTuple(torch.nn.Module):
    def forward(self, x):
        return (2 * x, [x + x])
    

for ModelCls in (Model, ModelWithTuple, ModelWithNestedTuple):
  model = ModelCls()
  inputs = torch.ones(5)
  torch_mlir.do(model, inputs, output_type="torch")


torch_mlir.do(model, inputs, output_type="tosa")
torch_mlir.do(model, inputs, output_type="tosa", dtype=torch.bfloat16)
torch_mlir.do(model, inputs, output_type="tosa", dtype=torch.bfloat16, output_prefix="out")
