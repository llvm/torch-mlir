# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ClassAnnotator, ImportOptions, ModuleBuilder

from utils import create_script_function

import typing

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


# CHECK-LABEL:   func.func @__torch__.prim_NumToTensor(
# CHECK-SAME:                           %[[ARG:.*]]: !torch.int) -> !torch.tensor {
# CHECK:           %[[RET:.*]] = torch.prim.NumToTensor.Scalar %[[ARG]] : !torch.int -> !torch.tensor
# CHECK:           return %[[RET]] : !torch.tensor
# CHECK:         }
@mb.import_function
@torch.jit.script
def prim_NumToTensor(i: int):
    return _to_tensor(i)


# CHECK-LABEL:   func.func @__torch__.prim_Print(
# CHECK-SAME:                     %[[ARG:.*]]: !torch.tensor) -> !torch.none {
# CHECK:           %[[STR:.*]] = torch.constant.str "x"
# CHECK:           torch.prim.Print(%[[STR]], %[[ARG]]) : !torch.str, !torch.tensor
@mb.import_function
@torch.jit.script
def prim_Print(x):
    print("x", x)


# CHECK-LABEL:   func.func @__torch__.prim_RaiseException() -> !torch.none {
# CHECK:           %[[ERRORSTR:.*]] = torch.constant.str "Error"
# CHECK:           %[[NONE:.*]] = torch.prim.Uninitialized : !torch.none
# CHECK:           torch.prim.RaiseException %[[ERRORSTR]]
# CHECK:           return %[[NONE]] : !torch.none
@mb.import_function
@torch.jit.script
def prim_RaiseException():
    raise Exception("Error")


# CHECK-LABEL:   func.func @__torch__.prim_unchecked_cast(
# CHECK-SAME:                              %[[ARG:.*]]: !torch.optional<int>) -> !torch.int {
# CHECK:           %[[NONE:.*]] = torch.constant.none
# CHECK:           %[[C3:.*]] = torch.constant.int 3
# CHECK:           %[[IS_NONE:.*]] = torch.aten.__is__ %[[ARG]], %[[NONE]] : !torch.optional<int>, !torch.none -> !torch.bool
# CHECK:           %[[RESULT:.*]] = torch.prim.If %[[IS_NONE]] -> (!torch.int) {
# CHECK:             torch.prim.If.yield %[[C3]] : !torch.int
# CHECK:           } else {
# CHECK:             %[[CASTED:.*]] = torch.prim.unchecked_cast %[[ARG]] : !torch.optional<int> -> !torch.int
# CHECK:             torch.prim.If.yield %[[CASTED]] : !torch.int
# CHECK:           }
# CHECK:           return %[[RESULT:.*]] : !torch.int
@mb.import_function
@torch.jit.script
def prim_unchecked_cast(i: typing.Optional[int]):
    if i is None:
        return 3
    return i


# CHECK-LABEL:   func.func @__torch__.prim_TupleUnpack(
# CHECK-SAME:                     %[[ARG:.*]]: !torch.tuple<int, int>) -> !torch.int {
# CHECK:           %[[RET:.*]]:2 = torch.prim.TupleUnpack %[[ARG]] : !torch.tuple<int, int> -> !torch.int, !torch.int
# CHECK:           return %[[RET]]#0 : !torch.int
@mb.import_function
@torch.jit.script
def prim_TupleUnpack(tup: typing.Tuple[int, int]):
    val, _ = tup
    return val


# CHECK-LABEL:   func.func @__torch__.prim_TupleIndex(
# CHECK-SAME:                     %[[ARG:.*]]: !torch.tuple<tensor, tensor>) -> !torch.tensor {
# CHECK:           %[[RET:.*]] = torch.prim.TupleIndex %[[ARG]], %[[IDX:.*]] : !torch.tuple<tensor, tensor>, !torch.int -> !torch.tensor
# CHECK:           return %[[RET]] : !torch.tensor
@mb.import_function
@torch.jit.script
def prim_TupleIndex(tup: typing.Tuple[torch.Tensor, torch.Tensor]):
    return tup[0]


# CHECK-LABEL:   func.func @__torch__.prim_ListUnpack(
# CHECK-SAME:                     %[[ARG:.*]]: !torch.list<int>) -> !torch.int {
# CHECK:           %[[RET:.*]]:3 = torch.prim.ListUnpack %[[ARG]] : !torch.list<int> -> !torch.int, !torch.int
# CHECK:           return %[[RET]]#1 : !torch.int
@mb.import_function
@torch.jit.script
def prim_ListUnpack(l: typing.List[int]):
    _, val, _ = l
    return val


# CHECK-LABEL:   func.func @__torch__.prim_dtype(
# CHECK-SAME:                               %[[ARG:.*]]: !torch.tensor) -> !torch.int {
# CHECK:           %[[RET:.*]] = torch.prim.dtype %[[ARG]] : !torch.tensor -> !torch.int
# CHECK:           return %[[RET]] : !torch.int
@mb.import_function
@torch.jit.script
def prim_dtype(x):
    return x.dtype


# CHECK-LABEL:   func.func @__torch__.prim_layout(
# CHECK-SAME:                                %[[ARG:.*]]: !torch.tensor) -> !torch.int {
# CHECK:           %[[RET:.*]] = torch.prim.layout %[[ARG]] : !torch.tensor -> !torch.int
# CHECK:           return %[[RET]] : !torch.int
@mb.import_function
@torch.jit.script
def prim_layout(x):
    return x.layout


# CHECK-LABEL:   func.func @__torch__.prim_device(
# CHECK-SAME:                                %[[ARG:.*]]: !torch.tensor) -> !torch.Device {
# CHECK:           %[[RET:.*]] = torch.prim.device %[[ARG]] : !torch.tensor -> !torch.Device
# CHECK:           return %[[RET]] : !torch.Device
@mb.import_function
@torch.jit.script
def prim_device(x):
    return x.device


# CHECK-LABEL:   func.func @__torch__.prim_min(
# CHECK-SAME:                             %[[ARG:.*]]: !torch.int) -> !torch.tuple<int, int, int> {
# CHECK:           %[[SINGLETON:.*]] = torch.prim.ListConstruct %[[ARG]] : (!torch.int) -> !torch.list<int>
# CHECK:           %[[MIN1:.*]] = torch.prim.min.self_int %[[SINGLETON]] : !torch.list<int> -> !torch.int
# CHECK:           %[[MIN2:.*]] = torch.prim.min.int %[[ARG]], %[[ARG]] : !torch.int, !torch.int -> !torch.int
# CHECK:           %[[ARG_3_TIMES:.*]] = torch.prim.ListConstruct %[[ARG]], %[[ARG]], %[[ARG]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
# CHECK:           %[[MIN3:.*]] = torch.prim.min.self_int %[[ARG_3_TIMES]] : !torch.list<int> -> !torch.int
# CHECK:           %[[RET:.*]] = torch.prim.TupleConstruct %[[MIN1]], %[[MIN2]], %[[MIN3]] : !torch.int, !torch.int, !torch.int
# CHECK:           return %[[RET]] : !torch.tuple<int, int, int>
@mb.import_function
@torch.jit.script
def prim_min(x: int):
    return min(x), min(x, x), min(x, x, x)


# CHECK-LABEL:   func.func @__torch__.prim_max(
# CHECK-SAME:                             %[[ARG:.*]]: !torch.int) -> !torch.tuple<int, int, int> {
# CHECK:           %[[SINGLETON:.*]] = torch.prim.ListConstruct %[[ARG]] : (!torch.int) -> !torch.list<int>
# CHECK:           %[[MAX1:.*]] = torch.prim.max.self_int %[[SINGLETON]] : !torch.list<int> -> !torch.int
# CHECK:           %[[MAX2:.*]] = torch.prim.max.int %[[ARG]], %[[ARG]] : !torch.int, !torch.int -> !torch.int
# CHECK:           %[[ARG_3_TIMES:.*]] = torch.prim.ListConstruct %[[ARG]], %[[ARG]], %[[ARG]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
# CHECK:           %[[MAX3:.*]] = torch.prim.max.self_int %[[ARG_3_TIMES]] : !torch.list<int> -> !torch.int
# CHECK:           %[[RET:.*]] = torch.prim.TupleConstruct %[[MAX1]], %[[MAX2]], %[[MAX3]] : !torch.int, !torch.int, !torch.int
# CHECK:           return %[[RET]] : !torch.tuple<int, int, int>
@mb.import_function
@torch.jit.script
def prim_max(x: int):
    return max(x), max(x, x), max(x, x, x)


# CHECK-LABEL:   func.func @__torch__.prim_Constant_list() -> !torch.list<int> {
# CHECK:           %[[A:.*]] = torch.constant.int 1
# CHECK:           %[[B:.*]] = torch.constant.int 2
# CHECK:           %[[C:.*]] = torch.constant.int 3
# CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %[[A]], %[[B]], %[[C]] :
# CHECK-SAME:          (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
# CHECK:           return %[[RET]] : !torch.list<int>
mb.import_function(
    create_script_function(
        "__torch__.prim_Constant_list",
        """
graph():
  %list : int[] = prim::Constant[value=[1, 2, 3]]()
  return (%list)
""",
    )
)

mb.module.operation.print()
print()

# CHECK-LABEL:   func.func @__torch__.prim_Constant_scalar() -> !torch.number {
# CHECK:           %[[A:.*]] = torch.tensor.literal
# CHECK:           %[[RET:.*]] = torch.aten.ScalarImplicit
# CHECK:           return %[[RET]] : !torch.number
import_options = ImportOptions()
import_options.assumeTensorsHaveValueSemantics = False
mb.import_function(
    create_script_function(
        "__torch__.prim_Constant_scalar",
        """
graph():
  %0 : Long(requires_grad=0, device=cpu) = prim::Constant[value={1}]()
  %1 : Scalar = aten::ScalarImplicit(%0)
  return (%1)
""",
        parse_tensor_constants=True,
    ),
    import_options,
)

mb.module.operation.print()
print()

# CHECK-LABEL:   func.func @__torch__.prim_Constant_scalar_value_semantics() -> !torch.number {
# CHECK:           %[[A:.*]] = torch.vtensor.literal
# CHECK:           %[[RET:.*]] = torch.aten.ScalarImplicit
# CHECK:           return %[[RET]] : !torch.number
import_options.assumeTensorsHaveValueSemantics = True
mb.import_function(
    create_script_function(
        "__torch__.prim_Constant_scalar_value_semantics",
        """
graph():
  %0 : Long(requires_grad=0, device=cpu) = prim::Constant[value={1}]()
  %1 : Scalar = aten::ScalarImplicit(%0)
  return (%1)
""",
        parse_tensor_constants=True,
    ),
    import_options,
)

mb.module.operation.print()
print()
