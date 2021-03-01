# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

import typing

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

# CHECK-LABEL:   func @prim_unchecked_cast(
# CHECK-SAME:                              %[[VAL_0:.*]]: !torch.optional<i64>) -> i64 {
# CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[C3:.*]] = constant 3 : i64
# CHECK:           %[[IS_NONE:.*]] = torch.kernel_call "aten::__is__" %[[VAL_0]], %[[NONE]] : (!torch.optional<i64>, !basicpy.NoneType) -> !basicpy.BoolType
# CHECK:           %[[COND:.*]] = basicpy.bool_cast %[[IS_NONE]] : !basicpy.BoolType -> i1
# CHECK:           %[[RESULT:.*]] = scf.if %[[COND]] -> (i64) {
# CHECK:             scf.yield %[[C3]] : i64
# CHECK:           } else {
# CHECK:             %[[CASTED:.*]] = torch.prim.unchecked_cast %[[VAL_0]] : !torch.optional<i64> -> i64
# CHECK:             scf.yield %[[CASTED]] : i64
# CHECK:           }
# CHECK:           return %[[RESULT:.*]] : i64
@mb.import_function
@torch.jit.script
def prim_unchecked_cast(i: typing.Optional[int]):
    if i is None:
        return 3
    return i

mb.module.operation.print()
print()
