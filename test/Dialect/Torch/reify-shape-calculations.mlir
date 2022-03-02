// RUN: torch-mlir-opt -torch-reify-shape-calculations -split-input-file %s | FileCheck %s

// CHECK: module {
// CHECK: func private @__torch_mlir_shape_fn.aten.tanh(

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate  {
// CHECK:             %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[TANH]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[SHAPE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[RESULT_SHAPE:.*]] = call @__torch_mlir_shape_fn.aten.tanh(%[[SHAPE]]) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[RESULT_SHAPE]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func @basic(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// torch.aten.add.Tensor also has a shape function that calls a "broadcast"
// helper, so this test also checks our logic for transitively pulling in
// callees of the shape functions.

// CHECK: module {
// CHECK: func private @__torch_mlir_shape_fn.aten.add.Tensor(

// CHECK-LABEL:   func @scalar_args(
// CHECK-SAME:                      %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                      %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate  {
// CHECK:             %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[INT1]] : !torch.vtensor, !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[ADD]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[ARG0_SHAPE:.*]] = torch.aten.size %[[ARG0]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[ARG1_SHAPE:.*]] = torch.aten.size %[[ARG1]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[SCALAR_CONVERTED:.*]] = torch.aten.Float.Scalar %[[INT1]] : !torch.int -> !torch.float
// CHECK:             %[[RESULT_SHAPE:.*]] = call @__torch_mlir_shape_fn.aten.add.Tensor(%[[ARG0_SHAPE]], %[[ARG1_SHAPE]], %[[SCALAR_CONVERTED]]) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.float) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[RESULT_SHAPE]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func @scalar_args(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor, !torch.vtensor, !torch.int -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK: module {
// CHECK: func private @__torch_mlir_shape_fn.aten.topk(

// CHECK-LABEL:   func @multiple_results(
// CHECK-SAME:                           %[[ARG:.*]]: !torch.tensor) -> (!torch.tensor, !torch.tensor) {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULTS:.*]]:2 = torch.shape.calculate  {
// CHECK:             %[[TOP_VALUES:.*]], %[[TOPK_INDICES:.*]] = torch.aten.topk %[[ARG]], %[[INT3]], %[[INT1]], %[[TRUE]], %[[TRUE]] : !torch.tensor, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.tensor, !torch.tensor
// CHECK:             torch.shape.calculate.yield %[[TOP_VALUES]], %[[TOPK_INDICES]] : !torch.tensor, !torch.tensor
// CHECK:           } shapes  {
// CHECK:             %[[ARG_SHAPE:.*]] = torch.aten.size %[[ARG]] : !torch.tensor -> !torch.list<!torch.int>
// CHECK:             %[[TOPK_SHAPE_TUPLE:.*]] = call @__torch_mlir_shape_fn.aten.topk(%[[ARG_SHAPE]], %[[INT3]], %[[INT1]], %[[TRUE]], %[[TRUE]]) : (!torch.list<!torch.int>, !torch.int, !torch.int, !torch.bool, !torch.bool) -> !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>>
// CHECK:             %[[TOPK_SHAPE:.*]]:2 = torch.prim.TupleUnpack %[[TOPK_SHAPE_TUPLE]] : !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>> -> !torch.list<!torch.int>, !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[TOPK_SHAPE]]#0, %[[TOPK_SHAPE]]#1 : !torch.list<!torch.int>, !torch.list<!torch.int>
// CHECK:           } : !torch.tensor, !torch.tensor
// CHECK:           return %[[RESULTS:.*]]#0, %[[RESULTS]]#1 : !torch.tensor, !torch.tensor

func @multiple_results(%arg0: !torch.tensor) -> (!torch.tensor, !torch.tensor) {
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.topk %arg0, %int3, %int1, %true, %true : !torch.tensor, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.tensor, !torch.tensor
  return %values, %indices : !torch.tensor, !torch.tensor
}

// -----

// CHECK-LABEL:   func @derefine_none(
// CHECK-SAME:                  %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                  %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate  {
// CHECK:             %[[CONV:.*]] = torch.aten.conv2d %[[ARG0]], %[[ARG1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[CONV]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[SHAPE0:.*]] = torch.aten.size %[[ARG0]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[SHAPE1:.*]] = torch.aten.size %[[ARG1]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[DEREFINED:.*]] = torch.derefine %{{.*}} : !torch.none to !torch.optional<!torch.list<!torch.int>>
// CHECK:             %[[SHAPE:.*]] = call @__torch_mlir_shape_fn.aten.conv2d(%[[SHAPE0]], %[[SHAPE1]], %[[DEREFINED]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.optional<!torch.list<!torch.int>>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func @derefine_none(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %24 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %25 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %26 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %29 = torch.aten.conv2d %arg0, %arg1, %none, %24, %25, %26, %int1 : !torch.vtensor, !torch.vtensor, !torch.none, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int -> !torch.vtensor
  return %29 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func @derefine_tensor(
// CHECK-SAME:                          %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[C1EM5:.*]] = torch.constant.float 1.000000e-05
// CHECK:           %[[C1EM1:.*]] = torch.constant.float 1.000000e-01
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[ARG]] : !torch.vtensor to !torch.optional<!torch.vtensor>
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate  {
// CHECK:             %[[BN:.*]] = torch.aten.batch_norm %[[ARG]], %[[DEREFINED]], %[[NONE]], %[[NONE]], %[[NONE]], %[[FALSE]], %[[C1EM1]], %[[C1EM5]], %[[TRUE]] : !torch.vtensor, !torch.optional<!torch.vtensor>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[BN]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[ARG_SIZE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[NONE2:.*]] = torch.constant.none
// CHECK:             %[[IS:.*]] = torch.aten.__is__ %[[DEREFINED]], %[[NONE2]] : !torch.optional<!torch.vtensor>, !torch.none -> !torch.bool
// CHECK:             %[[DEREFINED_OPTIONAL_SIZE:.*]] = torch.prim.If %[[IS]] -> (!torch.optional<!torch.list<!torch.int>>) {
// CHECK:               %[[DEREFINE_NONE:.*]] = torch.derefine %[[NONE2]] : !torch.none to !torch.optional<!torch.list<!torch.int>>
// CHECK:               torch.prim.If.yield %[[DEREFINE_NONE]] : !torch.optional<!torch.list<!torch.int>>
// CHECK:             } else {
// CHECK:               %[[DOWNCASTED:.*]] = torch.prim.unchecked_cast %[[DEREFINED]] : !torch.optional<!torch.vtensor> -> !torch.vtensor
// CHECK:               %[[DOWNCASTED_SIZE:.*]] = torch.aten.size %[[DOWNCASTED]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:               %[[DEREFINE_DOWNCASTED_SIZE:.*]] = torch.derefine %[[DOWNCASTED_SIZE]] : !torch.list<!torch.int> to !torch.optional<!torch.list<!torch.int>>
// CHECK:               torch.prim.If.yield %[[DEREFINE_DOWNCASTED_SIZE]] : !torch.optional<!torch.list<!torch.int>>
// CHECK:             }
// CHECK:             %[[DEREFINED_NONE1:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.list<!torch.int>>
// CHECK:             %[[DEREFINED_NONE2:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.list<!torch.int>>
// CHECK:             %[[DEREFINED_NONE3:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.list<!torch.int>>
// CHECK:             %[[VAL_20:.*]] = call @__torch_mlir_shape_fn.aten.batch_norm(%[[ARG_SIZE]], %[[DEREFINED_OPTIONAL_SIZE:.*]], %[[DEREFINED_NONE1]], %[[DEREFINED_NONE2]], %[[DEREFINED_NONE3]], %[[FALSE]], %[[C1EM1]], %[[C1EM5]], %[[TRUE]]) : (!torch.list<!torch.int>, !torch.optional<!torch.list<!torch.int>>, !torch.optional<!torch.list<!torch.int>>, !torch.optional<!torch.list<!torch.int>>, !torch.optional<!torch.list<!torch.int>>, !torch.bool, !torch.float, !torch.float, !torch.bool) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[VAL_20]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func @derefine_tensor(%arg0: !torch.vtensor) -> !torch.vtensor {
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %none = torch.constant.none
  %derefined_tensor = torch.derefine %arg0 : !torch.vtensor to !torch.optional<!torch.vtensor>
  %0 = torch.aten.batch_norm %arg0, %derefined_tensor, %none, %none, %none, %false, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor, !torch.optional<!torch.vtensor>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func @shape_function_list_operands(
// CHECK-SAME:                                       %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                       %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[ARG1]] : (!torch.vtensor) -> !torch.list<!torch.vtensor>
// CHECK:           %[[VAL_3:.*]] = torch.shape.calculate  {
// CHECK:             %[[VAL_4:.*]] = torch.aten.index.Tensor %[[ARG0]], %[[LIST]] : !torch.vtensor, !torch.list<!torch.vtensor> -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[VAL_4]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[ARG0_SHAPE:.*]] = torch.aten.size %[[ARG0]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[ADJUSTED_LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<!torch.optional<!torch.list<!torch.int>>>
// CHECK:             %[[LIST_SIZE:.*]] = torch.aten.len.t %[[LIST]] : !torch.list<!torch.vtensor> -> !torch.int
// CHECK:             %[[CTRUE:.*]] = torch.constant.bool true
// CHECK:             torch.prim.Loop %[[LIST_SIZE]], %[[CTRUE]], init()  {
// CHECK:             ^bb0(%[[ITER_NUM:.*]]: !torch.int):
// CHECK:               %[[UNADJUSTED_ELEMENT:.*]] = torch.aten.__getitem__.t %[[LIST]], %[[ITER_NUM]] : !torch.list<!torch.vtensor>, !torch.int -> !torch.vtensor
// CHECK:               %[[UNADJUSTED_ELEMENT_SHAPE:.*]] = torch.aten.size %[[UNADJUSTED_ELEMENT]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:               %[[ADJUSTED_ELEMENT:.*]] = torch.derefine %[[UNADJUSTED_ELEMENT_SHAPE]] : !torch.list<!torch.int> to !torch.optional<!torch.list<!torch.int>>
// CHECK:               %{{.*}} = torch.aten.append.t %[[ADJUSTED_LIST]], %[[ADJUSTED_ELEMENT]] : !torch.list<!torch.optional<!torch.list<!torch.int>>>, !torch.optional<!torch.list<!torch.int>> -> !torch.list<!torch.optional<!torch.list<!torch.int>>>
// CHECK:               torch.prim.Loop.condition %[[CTRUE]], iter()
// CHECK:             } : (!torch.int, !torch.bool) -> ()
// CHECK:             %[[RESULT_SHAPE:.*]] = call @__torch_mlir_shape_fn.aten.index.Tensor(%[[ARG0_SHAPE]], %[[ADJUSTED_LIST]]) : (!torch.list<!torch.int>, !torch.list<!torch.optional<!torch.list<!torch.int>>>) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[RESULT_SHAPE]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[VAL_15:.*]] : !torch.vtensor
func @shape_function_list_operands(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor) -> !torch.list<!torch.vtensor>
  %1 = torch.aten.index.Tensor %arg0, %0 : !torch.vtensor, !torch.list<!torch.vtensor> -> !torch.vtensor
  return %1 : !torch.vtensor
}
