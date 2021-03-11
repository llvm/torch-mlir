# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

import typing

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()


# CHECK-LABEL:   func @__torch__.prim_NumToTensor(
# CHECK-SAME:                           %[[ARG:.*]]: i64) -> !numpy.ndarray<*:!numpy.any_dtype> {
# CHECK:           %[[RET:.*]] = torch.prim.NumToTensor %[[ARG]] : i64 -> !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:           return %[[RET]] : !numpy.ndarray<*:!numpy.any_dtype>
# CHECK:         }

@mb.import_function
@torch.jit.script
def prim_NumToTensor(i: int):
    return _to_tensor(i)

# CHECK-LABEL:   func @__torch__.prim_Print(
# CHECK-SAME:                     %[[ARG:.*]]: !numpy.ndarray<*:!numpy.any_dtype>) -> !basicpy.NoneType {
# CHECK:           %[[STR:.*]] = basicpy.bytes_constant "x"
# CHECK:           torch.prim.Print(%[[STR]], %[[ARG]]) : !basicpy.BytesType, !numpy.ndarray<*:!numpy.any_dtype>
@mb.import_function
@torch.jit.script
def prim_Print(x):
    print("x", x)

# CHECK-LABEL:   func @__torch__.prim_RaiseException() -> !basicpy.NoneType {
# CHECK:           %[[ERRORSTR:.*]] = basicpy.bytes_constant "Error"
# CHECK:           %[[NONE:.*]] = torch.prim.Uninitialized : !basicpy.NoneType
# CHECK:           torch.prim.RaiseException %[[ERRORSTR]]
# CHECK:           return %[[NONE]] : !basicpy.NoneType
@mb.import_function
@torch.jit.script
def prim_RaiseException():
    raise Exception("Error")

# CHECK-LABEL:   func @__torch__.prim_unchecked_cast(
# CHECK-SAME:                              %[[ARG:.*]]: !torch.optional<i64>) -> i64 {
# CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
# CHECK:           %[[C3:.*]] = constant 3 : i64
# CHECK:           %[[IS_NONE:.*]] = torch.kernel_call "aten::__is__" %[[ARG]], %[[NONE]] : (!torch.optional<i64>, !basicpy.NoneType) -> !basicpy.BoolType
# CHECK:           %[[COND:.*]] = basicpy.bool_cast %[[IS_NONE]] : !basicpy.BoolType -> i1
# CHECK:           %[[RESULT:.*]] = scf.if %[[COND]] -> (i64) {
# CHECK:             scf.yield %[[C3]] : i64
# CHECK:           } else {
# CHECK:             %[[CASTED:.*]] = torch.prim.unchecked_cast %[[ARG]] : !torch.optional<i64> -> i64
# CHECK:             scf.yield %[[CASTED]] : i64
# CHECK:           }
# CHECK:           return %[[RESULT:.*]] : i64
@mb.import_function
@torch.jit.script
def prim_unchecked_cast(i: typing.Optional[int]):
    if i is None:
        return 3
    return i

# CHECK-LABEL:   func @__torch__.prim_TupleUnpack(
# CHECK-SAME:                     %[[ARG:.*]]: !basicpy.TupleType) -> i64 {
# CHECK:           %[[RET:.*]]:2 = torch.prim.TupleUnpack %[[ARG]] : !basicpy.TupleType -> i64, i64
# CHECK:           return %[[RET]]#0 : i64
@mb.import_function
@torch.jit.script
def prim_TupleUnpack(tup: typing.Tuple[int, int]):
    val, _ = tup
    return val

# CHECK-LABEL:   func @__torch__.prim_TupleIndex(
# CHECK-SAME:                     %[[ARG:.*]]: !basicpy.TupleType) -> i64 {
# CHECK:           %[[RET:.*]] = torch.prim.TupleIndex %[[ARG]], %[[IDX:.*]] : !basicpy.TupleType, i64 -> i64
# CHECK:           return %[[RET]] : i64
@mb.import_function
@torch.jit.script
def prim_TupleIndex(tup: typing.Tuple[int, int]):
    return tup[0]

# CHECK-LABEL:   func @__torch__.prim_ListUnpack(
# CHECK-SAME:                     %[[ARG:.*]]: !basicpy.ListType) -> i64 {
# CHECK:           %[[RET:.*]]:3 = torch.prim.ListUnpack %[[ARG]] : !basicpy.ListType -> i64, i64
# CHECK:           return %[[RET]]#1 : i64
@mb.import_function
@torch.jit.script
def prim_ListUnpack(l: typing.List[int]):
    _, val, _ = l
    return val

# CHECK-LABEL:   func @__torch__.prim_dtype(
# CHECK-SAME:                               %[[ARG:.*]]: !numpy.ndarray<*:!numpy.any_dtype>) -> i64 {
# CHECK:           %[[RET:.*]] = torch.prim.dtype %[[ARG]] : !numpy.ndarray<*:!numpy.any_dtype> -> i64
# CHECK:           return %[[RET]] : i64
@mb.import_function
@torch.jit.script
def prim_dtype(x):
    return x.dtype

# CHECK-LABEL:   func @__torch__.prim_device(
# CHECK-SAME:                                %[[ARG:.*]]: !numpy.ndarray<*:!numpy.any_dtype>) -> !torch.Device {
# CHECK:           %[[RET:.*]] = torch.prim.device %[[ARG]] : !numpy.ndarray<*:!numpy.any_dtype> -> !torch.Device
# CHECK:           return %[[RET]] : !torch.Device
@mb.import_function
@torch.jit.script
def prim_device(x):
    return x.device

mb.module.operation.print()
print()
