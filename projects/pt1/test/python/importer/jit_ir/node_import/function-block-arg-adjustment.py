# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

from torch_mlir.jit_ir_importer import ModuleBuilder

from utils import create_script_function

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# CHECK-LABEL:   func.func @__torch__.refined_block_arg(
# CHECK-SAME:                                      %[[ARG:.*]]: !torch.tensor) -> !torch.tensor {
# CHECK:           %[[REFINED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.tensor to !torch.tensor<[1,384],f32>
# CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[REFINED]] : !torch.tensor<[1,384],f32> to !torch.tensor
# CHECK:           return %[[RESULT]] : !torch.tensor
mb.import_function(
    create_script_function(
        "__torch__.refined_block_arg",
        """
graph(%0 : Float(1, 384)):
    return (%0)
""",
    )
)

mb.module.operation.print()
print()
