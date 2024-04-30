# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

import typing

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


# CHECK-LABEL:   func.func @__torch__.prim_Loop_forlike(
# CHECK-SAME:                            %[[MAX_ITERATIONS:.*]]: !torch.int) -> !torch.float {
# CHECK:           %[[BOOL_TRUE:.*]] = torch.constant.bool true
# CHECK:           %[[F_INIT:.*]] = torch.constant.float 0.000000e+00
# CHECK:           %[[RESULTS:.*]] = torch.prim.Loop %[[MAX_ITERATIONS]], %[[BOOL_TRUE]], init(%[[F_INIT]])  {
# CHECK:           ^bb0(%[[IV:.*]]: !torch.int, %[[F_ITER:.*]]: !torch.float):
# CHECK:             %[[F_NEXT:.*]] = torch.aten.add.float_int %[[F_ITER]], %[[IV]] : !torch.float, !torch.int -> !torch.float
# CHECK:             torch.prim.Loop.condition %[[BOOL_TRUE]], iter(%[[F_NEXT]] : !torch.float)
# CHECK:           } : (!torch.int, !torch.bool, !torch.float) -> !torch.float
# CHECK:           return %[[RESULTS:.*]] : !torch.float
@mb.import_function
@torch.jit.script
def prim_Loop_forlike(n: int):
    f = 0.0
    for i in range(n):
        f += i
    return f


# CHECK-LABEL:   func.func @__torch__.prim_Loop_whilelike(
# CHECK-SAME:                              %[[VAL_0:.*]]: !torch.int) -> !torch.float {
# CHECK:           %[[F_INIT:.*]] = torch.constant.float 3.200000e+00
# CHECK:           %[[MAX_ITERATIONS:.*]] = torch.constant.int 9223372036854775807
# CHECK:           %[[COND_INIT:.*]] = torch.aten.lt.float_int %[[F_INIT]], %[[VAL_0]] : !torch.float, !torch.int -> !torch.bool
# CHECK:           %[[RET:.*]] = torch.prim.Loop %[[MAX_ITERATIONS]], %[[COND_INIT]], init(%[[F_INIT]])  {
# CHECK:           ^bb0(%[[F_ITER:.*]]: !torch.int, %[[F_ITER:.*]]: !torch.float):
# CHECK:             %[[F_NEXT:.*]] = torch.aten.mul.float %[[F_ITER]], %[[F_ITER]] : !torch.float, !torch.float -> !torch.float
# CHECK:             %[[COND_ITER:.*]] = torch.aten.lt.float_int %[[F_NEXT]], %[[VAL_0]] : !torch.float, !torch.int -> !torch.bool
# CHECK:             torch.prim.Loop.condition %[[COND_ITER]], iter(%[[F_NEXT]] : !torch.float)
# CHECK:           } : (!torch.int, !torch.bool, !torch.float) -> !torch.float
# CHECK:           return %[[RET:.*]] : !torch.float
@mb.import_function
@torch.jit.script
def prim_Loop_whilelike(n: int):
    f = 3.2
    while f < n:
        f = f * f
    return f


# CHECK-LABEL:   func.func @__torch__.prim_Loop_derefine(
# CHECK-SAME:                             %[[ARG:.*]]: !torch.int) -> !torch.optional<int> {
# CHECK:           %[[TRUE:.*]] = torch.constant.bool true
# CHECK:           %[[NONE:.*]] = torch.constant.none
# CHECK:           %[[NONE_DEREFINED:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<int>
# CHECK:           %[[RET:.*]] = torch.prim.Loop %[[ARG]], %[[TRUE]], init(%[[NONE_DEREFINED]])  {
# CHECK:           ^bb0(%[[IV:.*]]: !torch.int, %[[X_ITER:.*]]: !torch.optional<int>):
# CHECK:             %[[X_NEXT:.*]] = torch.derefine %[[ARG]] : !torch.int to !torch.optional<int>
# CHECK:             torch.prim.Loop.condition %[[TRUE]], iter(%[[X_NEXT]] : !torch.optional<int>)
# CHECK:           } : (!torch.int, !torch.bool, !torch.optional<int>) -> !torch.optional<int>
# CHECK:           return %[[RET:.*]] : !torch.optional<int>
@mb.import_function
@torch.jit.script
def prim_Loop_derefine(n: int):
    x: typing.Optional[int] = None
    for i in range(n):
        x = n
    return x


mb.module.operation.print()
print()
