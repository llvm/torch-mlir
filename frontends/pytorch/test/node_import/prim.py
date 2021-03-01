# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()


# CHECK-LABEL:   func @prim_NumToTensor(
# CHECK-SAME:                           %[[ARG:.*]]: i64) -> !numpy.ndarray<*:!numpy.any_dtype> {
# CHECK:           %[[RET:.*]] = torch.prim.NumToTensor %[[ARG]] : i64 -> !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:           return %[[RET]] : !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:         }

@mb.import_function
@torch.jit.script
def prim_NumToTensor(i: int):
    return _to_tensor(i)

# CHECK-LABEL:   func @prim_Print(
# CHECK-SAME:                     %[[ARG:.*]]: !numpy.ndarray<*:!numpy.any_dtype>) -> !basicpy.NoneType {
# CHECK:           %[[STR:.*]] = basicpy.bytes_constant "x"
# CHECK:           torch.prim.Print(%[[STR]], %[[ARG]]) : !basicpy.BytesType, !numpy.ndarray<*:!numpy.any_dtype>
@mb.import_function
@torch.jit.script
def prim_Print(x):
    print("x", x)

# CHECK-LABEL:   func @prim_RaiseException() -> !basicpy.NoneType {
# CHECK:           %[[ERRORSTR:.*]] = basicpy.bytes_constant "Error"
# CHECK:           %[[NONE:.*]] = torch.prim.Uninitialized : !basicpy.NoneType
# CHECK:           torch.prim.RaiseException %[[ERRORSTR]]
# CHECK:           return %[[NONE]] : !basicpy.NoneType
@mb.import_function
@torch.jit.script
def prim_RaiseException():
    raise Exception("Error")

mb.module.operation.print()
print()
