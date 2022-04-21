# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# CHECK-LABEL:   func @__torch__.identity(
# CHECK-SAME:                             %[[ARG:.*]]: !torch.tensor<[7],f32>) -> !torch.tensor {
# CHECK:           %[[RET:.*]] = torch.derefine %[[ARG]] : !torch.tensor<[7],f32> to !torch.tensor
# CHECK:           return %[[RET]] : !torch.tensor
def identity(x):
    return x
mb.import_function(torch.jit.trace(identity, torch.ones(7)))

mb.module.operation.print()
print()
