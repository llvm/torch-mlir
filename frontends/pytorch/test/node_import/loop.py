# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

import typing

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL:   func @__torch__.prim_Loop_forlike(
# CHECK-SAME:                            %[[MAX_ITERATIONS:.*]]: i64) -> f64 {
# CHECK:           %[[BOOL_TRUE:.*]] = basicpy.bool_constant true
# CHECK:           %[[F_INIT:.*]] = constant 0.000000e+00 : f64
# CHECK:           %[[RESULTS:.*]] = torch.prim.Loop %[[MAX_ITERATIONS]], %[[BOOL_TRUE]], init(%[[F_INIT]])  {
# CHECK:           ^bb0(%[[IV:.*]]: i64, %[[F_ITER:.*]]: f64):
# CHECK:             %[[F_NEXT:.*]] = torch.aten.add.float_int %[[F_ITER]], %[[IV]] : f64, i64 -> f64
# CHECK:             torch.prim.Loop.condition %[[BOOL_TRUE]], iter(%[[F_NEXT]] : f64)
# CHECK:           } : (i64, !basicpy.BoolType, f64) -> f64
# CHECK:           return %[[RESULTS:.*]] : f64
@mb.import_function
@torch.jit.script
def prim_Loop_forlike(n: int):
    f = 0.0
    for i in range(n):
        f += i
    return f

# CHECK-LABEL:   func @__torch__.prim_Loop_whilelike(
# CHECK-SAME:                              %[[VAL_0:.*]]: i64) -> f64 {
# CHECK:           %[[F_INIT:.*]] = constant 3.200000e+00 : f64
# CHECK:           %[[MAX_ITERATIONS:.*]] = constant 9223372036854775807 : i64
# CHECK:           %[[COND_INIT:.*]] = torch.aten.lt.float_int %[[F_INIT]], %[[VAL_0]] : f64, i64 -> !basicpy.BoolType
# CHECK:           %[[RET:.*]] = torch.prim.Loop %[[MAX_ITERATIONS]], %[[COND_INIT]], init(%[[F_INIT]])  {
# CHECK:           ^bb0(%[[F_ITER:.*]]: i64, %[[F_ITER:.*]]: f64):
# CHECK:             %[[F_NEXT:.*]] = torch.aten.mul.float %[[F_ITER]], %[[F_ITER]] : f64, f64 -> f64
# CHECK:             %[[COND_ITER:.*]] = torch.aten.lt.float_int %[[F_NEXT]], %[[VAL_0]] : f64, i64 -> !basicpy.BoolType
# CHECK:             torch.prim.Loop.condition %[[COND_ITER]], iter(%[[F_NEXT]] : f64)
# CHECK:           } : (i64, !basicpy.BoolType, f64) -> f64
# CHECK:           return %[[RET:.*]] : f64
@mb.import_function
@torch.jit.script
def prim_Loop_whilelike(n: int):
    f = 3.2
    while f < n:
        f = f * f
    return f

# CHECK-LABEL:   func @__torch__.prim_Loop_derefine(
# CHECK-SAME:                             %[[ARG:.*]]: i64) -> !torch.optional<i64> {
# CHECK:           %[[TRUE:.*]] = basicpy.bool_constant true
# CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[NONE_DEREFINED:.*]] = torch.derefine %[[NONE]] : !basicpy.NoneType to !torch.optional<i64>
# CHECK:           %[[RET:.*]] = torch.prim.Loop %[[ARG]], %[[TRUE]], init(%[[NONE_DEREFINED]])  {
# CHECK:           ^bb0(%[[IV:.*]]: i64, %[[X_ITER:.*]]: !torch.optional<i64>):
# CHECK:             %[[X_NEXT:.*]] = torch.derefine %[[ARG]] : i64 to !torch.optional<i64>
# CHECK:             torch.prim.Loop.condition %[[TRUE]], iter(%[[X_NEXT]] : !torch.optional<i64>)
# CHECK:           } : (i64, !basicpy.BoolType, !torch.optional<i64>) -> !torch.optional<i64>
# CHECK:           return %[[RET:.*]] : !torch.optional<i64>
@mb.import_function
@torch.jit.script
def prim_Loop_derefine(n: int):
    x: typing.Optional[int] = None
    for i in range(n):
        x = n
    return x

mb.module.operation.print()
print()
