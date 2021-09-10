# RUN: %PYTHON %s

import mlir.ir
from mlir.dialects import torch

with mlir.ir.Context() as ctx:
  torch.register_torch_dialect(ctx)
