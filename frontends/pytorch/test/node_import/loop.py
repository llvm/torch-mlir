# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL:   func @prim_Loop_forlike(
# CHECK-SAME:                            %[[MAX_ITERATIONS:.*]]: i64) -> f64 {
# CHECK:           %[[BOOL_TRUE:.*]] = basicpy.bool_constant true
# CHECK:           %[[F_INIT:.*]] = constant 0.000000e+00 : f64
# CHECK:           %[[RESULTS:.*]] = torch.prim.Loop %[[MAX_ITERATIONS]], %[[BOOL_TRUE]], init(%[[F_INIT]])  {
# CHECK:           ^bb0(%[[IV:.*]]: i64, %[[F_ITER:.*]]: f64):
# CHECK:             %[[F_NEXT:.*]] = torch.kernel_call "aten::add" %[[F_ITER]], %[[IV]] : (f64, i64) -> f64 {sigArgTypes = ["float", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["float"]}
# CHECK:             torch.prim.Loop.condition %[[BOOL_TRUE]] iter(%[[F_NEXT]]) : !basicpy.BoolType, (f64)
# CHECK:           } : (i64, !basicpy.BoolType, f64) -> f64
# CHECK:           return %[[RESULTS:.*]] : f64
@mb.import_function
@torch.jit.script
def prim_Loop_forlike(n: int):
    f = 0.0
    for i in range(n):
        f += i
    return f

# CHECK-LABEL:   func @prim_Loop_whilelike(
# CHECK-SAME:                              %[[VAL_0:.*]]: i64) -> f64 {
# CHECK:           %[[F_INIT:.*]] = constant 3.200000e+00 : f64
# CHECK:           %[[MAX_ITERATIONS:.*]] = constant 9223372036854775807 : i64
# CHECK:           %[[COND_INIT:.*]] = torch.kernel_call "aten::lt" %[[F_INIT]], %[[VAL_0]] : (f64, i64) -> !basicpy.BoolType {sigArgTypes = ["float", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["bool"]}
# CHECK:           %[[IV:.*]] = torch.prim.Loop %[[MAX_ITERATIONS]], %[[COND_INIT]], init(%[[F_INIT]])  {
# CHECK:           ^bb0(%[[F_ITER:.*]]: i64, %[[F_ITER:.*]]: f64):
# CHECK:             %[[F_NEXT:.*]] = torch.kernel_call "aten::mul" %[[F_ITER]], %[[F_ITER]] : (f64, f64) -> f64 {sigArgTypes = ["float", "float"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["float"]}
# CHECK:             %[[COND_ITER:.*]] = torch.kernel_call "aten::lt" %[[F_NEXT]], %[[VAL_0]] : (f64, i64) -> !basicpy.BoolType {sigArgTypes = ["float", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["bool"]}
# CHECK:             torch.prim.Loop.condition %[[COND_ITER]] iter(%[[F_NEXT]]) : !basicpy.BoolType, (f64)
# CHECK:           } : (i64, !basicpy.BoolType, f64) -> f64
# CHECK:           return %[[VAL_9:.*]] : f64
@mb.import_function
@torch.jit.script
def prim_Loop_whilelike(n: int):
    f = 3.2
    while f < n:
        f = f * f
    return f

mb.module.operation.print()
print()
