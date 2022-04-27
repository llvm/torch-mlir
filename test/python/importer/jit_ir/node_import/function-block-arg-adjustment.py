# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

from torch._C import CompilationUnit


# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

# Import TorchScript IR string as ScriptFunction.
def create_script_function(func_name, ts_ir_str):
    cu = CompilationUnit()
    return cu.create_function(func_name, torch._C.parse_ir(ts_ir_str))

# CHECK-LABEL:   func @__torch__.refined_block_arg(
# CHECK-SAME:                                      %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
# CHECK:           %[[REFINED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.tensor to !torch.tensor<[1,384],f32>
# CHECK:           %[[RESULT:.*]] = torch.derefine %[[REFINED]] : !torch.tensor<[1,384],f32> to !torch.tensor
# CHECK:           return %[[RESULT]] : !torch.tensor
script_function = create_script_function('__torch__.refined_block_arg', '''
graph(%0 : Float(1, 384)):
    return (%0)
''')

mb = ModuleBuilder()
mb.import_function(script_function)

mb.module.operation.print()
print()
