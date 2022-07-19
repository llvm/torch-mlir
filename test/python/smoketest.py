# RUN: %PYTHON %s

import torch_mlir.ir
from torch_mlir.dialects import torch

with torch_mlir.ir.Context() as ctx:
  torch.register_required_dialects(ctx)
