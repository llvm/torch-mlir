# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

from typing import Union

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# CHECK-LABEL:   func.func @__torch__.f(
# CHECK-SAME:                      %{{.*}}: !torch.union<float, int>) -> !torch.none {


@mb.import_function
@torch.jit.script
def f(x: Union[int, float]):
    return


assert isinstance(f, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
