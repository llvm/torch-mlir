# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL: @__torch__.prim_If(
# CHECK-SAME:           %[[B:.*]]: !basicpy.BoolType,
# CHECK-SAME:           %[[I:.*]]: i64) -> i64 {
@mb.import_function
@torch.jit.script
def prim_If(b: bool, i: int):
    # CHECK:           %[[I1:.*]] = basicpy.bool_cast %[[B]] : !basicpy.BoolType -> i1
    # CHECK:           %[[RES:.*]] = scf.if %[[I1]] -> (i64) {
    # CHECK:             %[[ADD:.*]] = torch.kernel_call "aten::add" %[[I]], %[[I]]
    # CHECK:             scf.yield %[[ADD]] : i64
    # CHECK:           } else {
    # CHECK:             %[[MUL:.*]] = torch.kernel_call "aten::mul" %[[I]], %[[I]]
    # CHECK:             scf.yield %[[MUL]] : i64
    # CHECK:           }
    # CHECK:           return %[[RES:.*]] : i64
    if b:
        return i + i
    else:
        return i * i
    # elif is modeled as a nested if, so no need to specially test it here.

# CHECK-LABEL:   func @__torch__.prim_If_derefine(
# CHECK-SAME:                           %[[B:.*]]: !basicpy.BoolType,
# CHECK-SAME:                           %[[I:.*]]: i64) -> !torch.optional<i64> {
# CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[PRED:.*]] = basicpy.bool_cast %[[B]] : !basicpy.BoolType -> i1
# CHECK:           %[[RES:.*]] = scf.if %[[PRED]] -> (!torch.optional<i64>) {
# CHECK:             %[[NONE_DEREFINED:.*]] = torch.derefine %[[NONE]] : !basicpy.NoneType -> !torch.optional<i64>
# CHECK:             scf.yield %[[NONE_DEREFINED]] : !torch.optional<i64>
# CHECK:           } else {
# CHECK:             %[[I_DEREFINED:.*]] = torch.derefine %[[I]] : i64 -> !torch.optional<i64>
# CHECK:             scf.yield %[[I_DEREFINED]] : !torch.optional<i64>
# CHECK:           }
# CHECK:           return %[[RES:.*]] : !torch.optional<i64>
@mb.import_function
@torch.jit.script
def prim_If_derefine(b: bool, i: int):
    if b:
        return None
    return i

mb.module.operation.print()
print()
