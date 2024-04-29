# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder
from typing import Tuple, Optional, NamedTuple

from utils import create_script_function

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()
NT = NamedTuple("NT", [("f1", Optional[torch.Tensor]), ("f2", Optional[torch.Tensor])])


# CHECK-LABEL:   func.func @__torch__.tuple(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) ->
# CHECK-SAME:            !torch.tuple<tensor, tensor> {
# CHECK:           %[[RET:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]] :
# CHECK-SAME:            !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
# CHECK:           return %[[RET]] : !torch.tuple<tensor, tensor>
@mb.import_function
@torch.jit.script
def tuple(t0, t1):
    return t0, t1


# CHECK-LABEL:   func.func @__torch__.tuple_optional(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) ->
# CHECK-SAME:            !torch.tuple<optional<tensor>, optional<tensor>> {
# CHECK:           %[[TNEW:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]] :
# CHECK-SAME:           !torch.tensor, !torch.tensor -> !torch.tuple<tensor, tensor>
# CHECK:           %[[RET:.*]] = torch.derefine %[[TNEW]] :
# CHECK-SAME:           !torch.tuple<tensor, tensor> to
# CHECK-SAME:           !torch.tuple<optional<tensor>, optional<tensor>>
# CHECK:           return %[[RET]] : !torch.tuple<optional<tensor>, optional<tensor>>
@mb.import_function
@torch.jit.script
def tuple_optional(t0, t1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    return t0, t1


# CHECK-LABEL:   func.func @__torch__.namedtuple_optional(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) ->
# CHECK-SAME:            !torch.tuple<optional<tensor>, optional<tensor>> {
# CHECK:           %[[RET:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]] :
# CHECK-SAME:            !torch.tensor, !torch.tensor ->
# CHECK-SAME:            !torch.tuple<optional<tensor>, optional<tensor>>
# CHECK:           return %[[RET]] : !torch.tuple<optional<tensor>, optional<tensor>>
@mb.import_function
@torch.jit.script
def namedtuple_optional(
    t0, t1
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    return NT(t0, t1)


# CHECK-LABEL:   func.func @__torch__.tuple_construct_arg_needs_refinement(
# CHECK-SAME:                                                         %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:                                                         %[[T1:.*]]: !torch.tensor) -> !torch.tuple<tensor, tensor> {
# CHECK:           %[[T0_REFINED:.*]] = torch.tensor_static_info_cast %[[T1]] : !torch.tensor to !torch.tensor<[4],f32>
# CHECK:           %[[T1_REFINED:.*]] = torch.tensor_static_info_cast %[[T0]] : !torch.tensor to !torch.tensor<[3],f32>
# CHECK:           %[[TUPLE:.*]] = torch.prim.TupleConstruct %[[T0_REFINED]], %[[T1_REFINED]] : !torch.tensor<[4],f32>, !torch.tensor<[3],f32> -> !torch.tuple<tensor<[4],f32>, tensor<[3],f32>>
# CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[TUPLE]] : !torch.tuple<tensor<[4],f32>, tensor<[3],f32>> to !torch.tuple<tensor, tensor>
# CHECK:           return %[[DEREFINED]] : !torch.tuple<tensor, tensor>
mb.import_function(
    create_script_function(
        "__torch__.tuple_construct_arg_needs_refinement",
        """
graph(%t0 : Tensor,
      %t1 : Tensor):
  %10 : (Float(4), Float(3)) = prim::TupleConstruct(%t1, %t0)
  return (%10)
""",
    )
)

mb.module.operation.print()
print()
