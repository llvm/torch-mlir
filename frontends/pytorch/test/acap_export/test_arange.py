# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s
torch_mlir.debug_trace_to_stderr()

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("arange_test", []) as f:
  x = torch.arange(10)
  f.returns([x])

# CHECK: %[[CST:.*]] = constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi64>
# CHECK: %[[R:.*]] = numpy.create_array_from_tensor %[[CST]]
# CHECK: return %[[R]]
mb.module.operation.print()
