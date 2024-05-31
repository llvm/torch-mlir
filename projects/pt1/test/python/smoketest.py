# RUN: %PYTHON %s

import torch_mlir.ir
from torch_mlir.dialects import torch

with torch_mlir.ir.Context() as ctx:
    torch.register_dialect(ctx)
    with torch_mlir.ir.Location.unknown() as loc:
        module = torch_mlir.ir.Module.create(loc)
        with torch_mlir.ir.InsertionPoint.at_block_begin(module.body):
            n = torch.ConstantNoneOp()
        module.operation.print()
