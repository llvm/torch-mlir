# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# Note: The "if without else" case is handled by yielding None from the
# else branch and making all defined values optional, so no special handling
# is needed.

# CHECK-LABEL: @__torch__.prim_If(
# CHECK-SAME:           %[[B:.*]]: !torch.bool,
# CHECK-SAME:           %[[I:.*]]: i64) -> i64 {
@mb.import_function
@torch.jit.script
def prim_If(b: bool, i: int):
    # CHECK:           %[[RES:.*]] = torch.prim.If %[[B]] -> (i64) {
    # CHECK:             %[[ADD:.*]] = torch.aten.add.int %[[I]], %[[I]]
    # CHECK:             torch.prim.If.yield %[[ADD]] : i64
    # CHECK:           } else {
    # CHECK:             %[[MUL:.*]] = torch.aten.mul.int %[[I]], %[[I]]
    # CHECK:             torch.prim.If.yield %[[MUL]] : i64
    # CHECK:           }
    # CHECK:           return %[[RES:.*]] : i64
    if b:
        return i + i
    else:
        return i * i

# CHECK-LABEL:   func @__torch__.prim_If_derefine(
# CHECK-SAME:                           %[[B:.*]]: !torch.bool,
# CHECK-SAME:                           %[[I:.*]]: i64) -> !torch.optional<i64> {
# CHECK:           %[[NONE:.*]] = torch.constant.none
# CHECK:           %[[RES:.*]] = torch.prim.If %[[B]] -> (!torch.optional<i64>) {
# CHECK:             %[[NONE_DEREFINED:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<i64>
# CHECK:             torch.prim.If.yield %[[NONE_DEREFINED]] : !torch.optional<i64>
# CHECK:           } else {
# CHECK:             %[[I_DEREFINED:.*]] = torch.derefine %[[I]] : i64 to !torch.optional<i64>
# CHECK:             torch.prim.If.yield %[[I_DEREFINED]] : !torch.optional<i64>
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
