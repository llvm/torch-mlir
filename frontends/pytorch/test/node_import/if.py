# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL: @f(
# CHECK-SAME:     %[[B:.*]]: !basicpy.BoolType,
# CHECK-SAME:     %[[I:.*]]: i64) -> i64 {
@mb.import_function
@torch.jit.script
def f(b: bool, i: int):
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

assert isinstance(f, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
