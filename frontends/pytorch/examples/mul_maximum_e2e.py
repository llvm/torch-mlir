# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import sys
import torch
import torch_mlir

lhs = torch.ones((4, 6, 1))
rhs = torch.ones((1, 1, 3)) * 0.6
bias = torch.ones((1, 1, 3)) * 0.2
threshold = torch.tensor((0.75, 0.25, 0.10))

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("mul_maximum", [lhs, rhs, threshold, bias]) as f:
  result = torch.maximum(lhs * rhs, threshold)
  result = result + bias
  f.returns([result])

print(f"Result(f{result.size()}) = {result}", file=sys.stderr)
# TODO: Currently need to route through:
#   npcomp-opt -aten-recognize-kernels -convert-aten-to-tcf \
#     -numpy-public-functions-to-tensor -canonicalize
mb.module.operation.print()
