// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.slice.strided$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.constant 0 : i64
// CHECK:         %[[T2:.*]] = arith.constant 2 : i64
// CHECK:         %[[T3:.*]] = arith.constant 10 : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T5:.*]] = arith.subi %[[C0_I64]], %[[T4]] : i64
// CHECK:         %[[T6:.*]] = arith.maxsi %[[T5]], %[[T1]] : i64
// CHECK:         %[[T7:.*]] = arith.minsi %[[T4]], %[[T6]] : i64
// CHECK:         %[[T8:.*]] = arith.addi %[[T4]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.cmpi sge, %[[T7]], %[[C0_I64]] : i64
// CHECK:         %[[T10:.*]] = arith.select %[[T9]], %[[T7]], %[[T8]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T11:.*]] = arith.subi %[[C0_I64_0]], %[[T4]] : i64
// CHECK:         %[[T12:.*]] = arith.maxsi %[[T11]], %[[T3]] : i64
// CHECK:         %[[T13:.*]] = arith.minsi %[[T4]], %[[T12]] : i64
// CHECK:         %[[T14:.*]] = arith.addi %[[T4]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.cmpi sge, %[[T13]], %[[C0_I64_0]] : i64
// CHECK:         %[[T16:.*]] = arith.select %[[T15]], %[[T13]], %[[T14]] : i64
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C0_1]] : tensor<?x?x?xf32>
// CHECK:         %[[T17:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_4:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[DIM_4]] : index to i64
// CHECK:         %[[C0_I64_5:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T20:.*]] = arith.cmpi eq, %[[T16]], %[[C0_I64_5]] : i64
// CHECK:         %[[T21:.*]] = arith.select %[[T20]], %[[T17]], %[[T16]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T10]], %[[C0_I64_5]], %[[C0_I64_5]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[T21]], %[[T18]], %[[T19]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_7:.*]] = tensor.from_elements %[[T2]], %[[C1_I64]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T22:.*]] = stablehlo.real_dynamic_slice %[[T0]], %[[FROM_ELEMENTS]], %[[FROM_ELEMENTS_6]], %[[FROM_ELEMENTS_7]] : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:         %[[T23:.*]] = torch_c.from_builtin_tensor %[[T22]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[T23]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.slice.strided$slice_like(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int10 = torch.constant.int 10
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int10, %int2 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.strided.static$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[2,65,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:         %[[T1:.*]] = arith.constant 0 : i64
// CHECK:         %[[T2:.*]] = arith.constant 2 : i64
// CHECK:         %[[T3:.*]] =  arith.constant 9223372036854775807 : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x65x256xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T5:.*]] = arith.subi %[[C0_I64]], %[[T4]] : i64
// CHECK:         %[[T6:.*]] = arith.maxsi %[[T5]], %[[T1]] : i64
// CHECK:         %[[T7:.*]] = arith.minsi %[[T4]], %[[T6]] : i64
// CHECK:         %[[T8:.*]] = arith.addi %[[T4]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.cmpi sge, %[[T7]], %[[C0_I64]] : i64
// CHECK:         %[[T10:.*]] = arith.select %[[T9]], %[[T7]], %[[T8]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T11:.*]] = arith.subi %[[C0_I64_0]], %[[T4]] : i64
// CHECK:         %[[T12:.*]] = arith.maxsi %[[T11]], %[[T3]] : i64
// CHECK:         %[[T13:.*]] = arith.minsi %[[T4]], %[[T12]] : i64
// CHECK:         %[[T14:.*]] = arith.addi %[[T4]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.cmpi sge, %[[T13]], %[[C0_I64_0]] : i64
// CHECK:         %[[T16:.*]] = arith.select %[[T15]], %[[T13]], %[[T14]] : i64
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C0_1]] : tensor<4x65x256xf32>
// CHECK:         %[[T17:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x65x256xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_4:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x65x256xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[DIM_4]] : index to i64
// CHECK:         %[[C0_I64_5:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T20:.*]] = arith.cmpi eq, %[[T16]], %[[C0_I64_5]] : i64
// CHECK:         %[[T21:.*]] = arith.select %[[T20]], %[[T17]], %[[T16]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T10]], %[[C0_I64_5]], %[[C0_I64_5]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[T21]], %[[T18]], %[[T19]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_7:.*]] = tensor.from_elements %[[T2]], %[[C1_I64]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T22:.*]] = stablehlo.real_dynamic_slice %[[T0]], %[[FROM_ELEMENTS]], %[[FROM_ELEMENTS_6]], %[[FROM_ELEMENTS_7]] : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x65x256xf32>
// CHECK:         %[[T23:.*]] = torch_c.from_builtin_tensor %[[T22]] : tensor<2x65x256xf32> -> !torch.vtensor<[2,65,256],f32>
// CHECK:         return %[[T23]] : !torch.vtensor<[2,65,256],f32>
func.func @torch.aten.slice.strided.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[2,65,256],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int9223372036854775807, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,65,256],f32>
  return %0 : !torch.vtensor<[2,65,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.last$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.constant 0 : i64
// CHECK:         %[[T2:.*]] = arith.constant 1 : i64
// CHECK:         %[[T3:.*]] = arith.constant -1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T5:.*]] = arith.subi %[[C0_I64]], %[[T4]] : i64
// CHECK:         %[[T6:.*]] = arith.maxsi %[[T5]], %[[T3]] : i64
// CHECK:         %[[T7:.*]] = arith.minsi %[[T4]], %[[T6]] : i64
// CHECK:         %[[T8:.*]] = arith.addi %[[T4]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.cmpi sge, %[[T7]], %[[C0_I64]] : i64
// CHECK:         %[[T10:.*]] = arith.select %[[T9]], %[[T7]], %[[T8]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T11:.*]] = arith.subi %[[C0_I64_0]], %[[T4]] : i64
// CHECK:         %[[T12:.*]] = arith.maxsi %[[T11]], %[[T1]] : i64
// CHECK:         %[[T13:.*]] = arith.minsi %[[T4]], %[[T12]] : i64
// CHECK:         %[[T14:.*]] = arith.addi %[[T4]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.cmpi sge, %[[T13]], %[[C0_I64_0]] : i64
// CHECK:         %[[T16:.*]] = arith.select %[[T15]], %[[T13]], %[[T14]] : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T17:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_3:.*]] = tensor.dim %[[T0]], %[[C1_2]] : tensor<?x?x?xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_4:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[DIM_4]] : index to i64
// CHECK:         %[[C0_I64_5:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T20:.*]] = arith.cmpi eq, %[[T16]], %[[C0_I64_5]] : i64
// CHECK:         %[[T21:.*]] = arith.select %[[T20]], %[[T18]], %[[T16]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[C0_I64_5]], %[[T10]], %[[C0_I64_5]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[T17]], %[[T21]], %[[T19]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_7:.*]] = tensor.from_elements %[[C1_I64]], %[[T2]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T22:.*]] = stablehlo.real_dynamic_slice %[[T0]], %[[FROM_ELEMENTS]], %[[FROM_ELEMENTS_6]], %[[FROM_ELEMENTS_7]] : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x1x?xf32>
// CHECK:         %[[T23:.*]] = torch_c.from_builtin_tensor %[[T22]] : tensor<?x1x?xf32> -> !torch.vtensor<[?,1,?],f32>
// CHECK:         return %[[T23]] : !torch.vtensor<[?,1,?],f32>
func.func @torch.aten.slice.last$slice_like(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,1,?],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int-1, %int0, %int1 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,1,?],f32>
  return %0 : !torch.vtensor<[?,1,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.last.static$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,1,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:         %[[T1:.*]] = arith.constant 0 : i64
// CHECK:         %[[T2:.*]] = arith.constant 1 : i64
// CHECK:         %[[T3:.*]] = arith.constant -1 : i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x65x256xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T5:.*]] = arith.subi %[[C0_I64]], %[[T4]] : i64
// CHECK:         %[[T6:.*]] = arith.maxsi %[[T5]], %[[T3]] : i64
// CHECK:         %[[T7:.*]] = arith.minsi %[[T4]], %[[T6]] : i64
// CHECK:         %[[T8:.*]] = arith.addi %[[T4]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.cmpi sge, %[[T7]], %[[C0_I64]] : i64
// CHECK:         %[[T10:.*]] = arith.select %[[T9]], %[[T7]], %[[T8]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T11:.*]] = arith.subi %[[C0_I64_0]], %[[T4]] : i64
// CHECK:         %[[T12:.*]] = arith.maxsi %[[T11]], %[[T1]] : i64
// CHECK:         %[[T13:.*]] = arith.minsi %[[T4]], %[[T12]] : i64
// CHECK:         %[[T14:.*]] = arith.addi %[[T4]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.cmpi sge, %[[T13]], %[[C0_I64_0]] : i64
// CHECK:         %[[T16:.*]] = arith.select %[[T15]], %[[T13]], %[[T14]] : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x65x256xf32>
// CHECK:         %[[T17:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_3:.*]] = tensor.dim %[[T0]], %[[C1_2]] : tensor<4x65x256xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_4:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x65x256xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[DIM_4]] : index to i64
// CHECK:         %[[C0_I64_5:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T20:.*]] = arith.cmpi eq, %[[T16]], %[[C0_I64_5]] : i64
// CHECK:         %[[T21:.*]] = arith.select %[[T20]], %[[T18]], %[[T16]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[C0_I64_5]], %[[T10]], %[[C0_I64_5]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[T17]], %[[T21]], %[[T19]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_7:.*]] = tensor.from_elements %[[C1_I64]], %[[T2]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T22:.*]] = stablehlo.real_dynamic_slice %[[T0]], %[[FROM_ELEMENTS]], %[[FROM_ELEMENTS_6]], %[[FROM_ELEMENTS_7]] : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x1x256xf32>
// CHECK:         %[[T23:.*]] = torch_c.from_builtin_tensor %[[T22]] : tensor<4x1x256xf32> -> !torch.vtensor<[4,1,256],f32>
// CHECK:         return %[[T23]] : !torch.vtensor<[4,1,256],f32>
func.func @torch.aten.slice.last.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,1,256],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int-1, %int0, %int1 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1,256],f32>
  return %0 : !torch.vtensor<[4,1,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.none$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = arith.constant 2 : i64
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C1_1]] : tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_3:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK:         %[[C0_I64_4:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T6:.*]] = arith.cmpi eq, %[[T2]], %[[C0_I64_4]] : i64
// CHECK:         %[[T7:.*]] = arith.select %[[T6]], %[[T4]], %[[T2]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[C0_I64_4]], %[[C0_I64]], %[[C0_I64_4]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_5:.*]] = tensor.from_elements %[[T3]], %[[T7]], %[[T5]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[C1_I64]], %[[T1]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T8:.*]] = stablehlo.real_dynamic_slice %[[T0]], %[[FROM_ELEMENTS]], %[[FROM_ELEMENTS]]_5, %[[FROM_ELEMENTS_6]] : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:         %[[T9:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[T9]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.slice.none$slice_like(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.aten.slice.Tensor %arg0, %int1, %none, %none, %int2 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.none.static$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,33,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = arith.constant 2 : i64
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x65x256xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x65x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C1_1]] : tensor<4x65x256xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_3:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x65x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[DIM_3]] : index to i64
// CHECK:         %[[C0_I64_4:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T6:.*]] = arith.cmpi eq, %[[T2]], %[[C0_I64_4]] : i64
// CHECK:         %[[T7:.*]] = arith.select %[[T6]], %[[T4]], %[[T2]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[C0_I64_4]], %[[C0_I64]], %[[C0_I64_4]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_5:.*]] = tensor.from_elements %[[T3]], %[[T7]], %[[T5]] : tensor<3xi64>
// CHECK:         %[[FROM_ELEMENTS_6:.*]] = tensor.from_elements %[[C1_I64]], %[[T1]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T8:.*]] = stablehlo.real_dynamic_slice %[[T0]], %[[FROM_ELEMENTS]], %[[FROM_ELEMENTS]]_5, %[[FROM_ELEMENTS_6]] : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x33x256xf32>
// CHECK:         %[[T9:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<4x33x256xf32> -> !torch.vtensor<[4,33,256],f32>
// CHECK:         return %[[T9]] : !torch.vtensor<[4,33,256],f32>
func.func @torch.aten.slice.none.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,33,256],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.aten.slice.Tensor %arg0, %int1, %none, %none, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,33,256],f32>
  return %0 : !torch.vtensor<[4,33,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[INT224:.*]] = torch.constant.int 224
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT]]-1, %[[INT]]224 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT224]]
// CHECK:         %[[T4:.*]] = shape.shape_of %[[T0]] : tensor<?x?x?x?xf32> -> tensor<4xindex>
// CHECK:         %[[T5:.*]] = shape.num_elements %[[T4]] : tensor<4xindex> -> index
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[T7:.*]] = arith.divui %[[T6]], %[[T3]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T7]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T8:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x?x?x?xf32>, tensor<2xi64>) -> tensor<?x224xf32>
// CHECK:         %[[T9:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<?x224xf32> -> !torch.vtensor<[?,224],f32>
// CHECK:         return %[[T9]] : !torch.vtensor<[?,224],f32>
func.func @torch.aten.view$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
  %int-1 = torch.constant.int -1
  %int224 = torch.constant.int 224
  %0 = torch.prim.ListConstruct %int-1, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,224],f32>
  return %1 : !torch.vtensor<[?,224],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.reshape$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?,?],f32> -> tensor<?x?x?x?x?xf32>
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[INT120:.*]] = torch.constant.int 120
// CHECK:         %[[INT4:.*]] = torch.constant.int 4
// CHECK:         %[[INT64:.*]] = torch.constant.int 64
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT]]-1, %[[INT]]120, %[[INT]]4, %[[INT]]64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT120]]
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %[[INT4]]
// CHECK:         %[[T5:.*]] = torch_c.to_i64 %[[INT64]]
// CHECK:         %[[T6:.*]] = shape.shape_of %[[T0]] : tensor<?x?x?x?x?xf32> -> tensor<5xindex>
// CHECK:         %[[T7:.*]] = shape.num_elements %[[T6]] : tensor<5xindex> -> index
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[T9:.*]] = arith.divui %[[T8]], %[[T3]] : i64
// CHECK:         %[[T10:.*]] = arith.divui %[[T9]], %[[T4]] : i64
// CHECK:         %[[T11:.*]] = arith.divui %[[T10]], %[[T5]] : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T11]], %[[T3]], %[[T4]], %[[T5]] : tensor<4xi64>
// CHECK:         %[[T12:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x?x?x?x?xf32>, tensor<4xi64>) -> tensor<?x120x4x64xf32>
// CHECK:         %[[T13:.*]] = torch_c.from_builtin_tensor %[[T12]] : tensor<?x120x4x64xf32> -> !torch.vtensor<[?,120,4,64],f32>
// CHECK:         return %[[T13]] : !torch.vtensor<[?,120,4,64],f32>
func.func @torch.aten.reshape$basic(%arg0: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
  %int-1 = torch.constant.int -1
  %int120 = torch.constant.int 120
  %int4 = torch.constant.int 4
  %int64 = torch.constant.int 64
  %0 = torch.prim.ListConstruct %int-1, %int120, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reshape %arg0, %0 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,120,4,64],f32>
  return %1 : !torch.vtensor<[?,120,4,64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view$to_rank1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = stablehlo.reshape %[[T0]] : (tensor<f32>) -> tensor<1xf32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[1],f32>
func.func @torch.aten.view$to_rank1(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1],f32>
  return %1 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.view$to_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1],f32> -> tensor<1xf32>
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:         %[[T2:.*]] = stablehlo.reshape %[[T0]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[],f32>
func.func @torch.aten.view$to_rank0(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim$0$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,1,2,1,2],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,1,2,1,2],f32> -> tensor<2x1x2x1x2xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch_c.from_builtin_tensor %[[T0]] : tensor<2x1x2x1x2xf32> -> !torch.vtensor<[2,1,2,1,2],f32>
// CHECK:         return %[[T1]] : !torch.vtensor<[2,1,2,1,2],f32>
func.func @torch.aten.squeeze.dim$0$static(%arg0: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,1,2,1,2],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.squeeze.dim %arg0, %int0 : !torch.vtensor<[2,1,2,1,2],f32>, !torch.int -> !torch.vtensor<[2,1,2,1,2],f32>
  return %0 : !torch.vtensor<[2,1,2,1,2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim$1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,?,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,1,?,1,?],f32> -> tensor<?x1x?x1x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C4]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[DIM]], %[[DIM_0]], %[[DIM_1]], %[[DIM_2]] : tensor<4xindex>
// CHECK:         %[[T5:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x1x?x1x?xf32>, tensor<4xindex>) -> tensor<?x?x1x?xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<?x?x1x?xf32> -> !torch.vtensor<[?,?,1,?],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[?,?,1,?],f32>
func.func @torch.aten.squeeze.dim$1(%arg0: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,?,1,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.squeeze.dim %arg0, %int1 : !torch.vtensor<[?,1,?,1,?],f32>, !torch.int -> !torch.vtensor<[?,?,1,?],f32>
  return %0 : !torch.vtensor<[?,?,1,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim$from_end(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,1,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,1,?,1,?],f32> -> tensor<?x1x?x1x?xf32>
// CHECK:         %[[INT:.*]]-2 = torch.constant.int -2
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C4]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[DIM]], %[[DIM_0]], %[[DIM_1]], %[[DIM_2]] : tensor<4xindex>
// CHECK:         %[[T5:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x1x?x1x?xf32>, tensor<4xindex>) -> tensor<?x1x?x?xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<?x1x?x?xf32> -> !torch.vtensor<[?,1,?,?],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[?,1,?,?],f32>
func.func @torch.aten.squeeze.dim$from_end(%arg0: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,1,?,?],f32> {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.squeeze.dim %arg0, %int-2 : !torch.vtensor<[?,1,?,1,?],f32>, !torch.int -> !torch.vtensor<[?,1,?,?],f32>
  return %0 : !torch.vtensor<[?,1,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.squeeze$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,2,2],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,1,2,1,2],f32> -> tensor<2x1x2x1x2xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<2x1x2x1x2xf32>
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<2x1x2x1x2xf32>
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C4]] : tensor<2x1x2x1x2xf32>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[DIM]], %[[DIM_0]], %[[DIM_1]] : tensor<3xindex>
// CHECK:         %[[T4:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<2x1x2x1x2xf32>, tensor<3xindex>) -> tensor<2x2x2xf32>
// CHECK:         %[[T5:.*]] = torch_c.from_builtin_tensor %[[T4]] : tensor<2x2x2xf32> -> !torch.vtensor<[2,2,2],f32>
// CHECK:         return %[[T5]] : !torch.vtensor<[2,2,2],f32>
func.func @torch.aten.squeeze$static(%arg0: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,2,2],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[2,1,2,1,2],f32> -> !torch.vtensor<[2,2,2],f32>
  return %0 : !torch.vtensor<[2,2,2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.unsqueeze$dim$0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1,?,?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[C1_I64]], %[[T1]], %[[T2]], %[[T3]], %[[T4]] : tensor<5xi64>
// CHECK:         %[[T5:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x?x?x?xf32>, tensor<5xi64>) -> tensor<1x?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<1x?x?x?x?xf32> -> !torch.vtensor<[1,?,?,?,?],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[1,?,?,?,?],f32>
func.func @torch.aten.unsqueeze$dim$0(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1,?,?,?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[1,?,?,?,?],f32>
  return %0 : !torch.vtensor<[1,?,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.unsqueeze$dim$1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,1,?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T1]], %[[C1_I64]], %[[T2]], %[[T3]], %[[T4]] : tensor<5xi64>
// CHECK:         %[[T5:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x?x?x?xf32>, tensor<5xi64>) -> tensor<?x1x?x?x?xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<?x1x?x?x?xf32> -> !torch.vtensor<[?,1,?,?,?],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[?,1,?,?,?],f32>
func.func @torch.aten.unsqueeze$dim$1(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,1,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.unsqueeze %arg0, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,1,?,?,?],f32>
  return %0 : !torch.vtensor<[?,1,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.unsqueeze$from_end(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT:.*]]-2 = torch.constant.int -2
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[DIM_2:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM_2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T1]], %[[T2]], %[[T3]], %[[C1_I64]], %[[T4]] : tensor<5xi64>
// CHECK:         %[[T5:.*]] = stablehlo.dynamic_reshape %[[T0]], %[[FROM_ELEMENTS]] : (tensor<?x?x?x?xf32>, tensor<5xi64>) -> tensor<?x?x?x1x?xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<?x?x?x1x?xf32> -> !torch.vtensor<[?,?,?,1,?],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[?,?,?,1,?],f32>
func.func @torch.aten.unsqueeze$from_end(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,1,?],f32> {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.unsqueeze %arg0, %int-2 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?,1,?],f32>
  return %0 : !torch.vtensor<[?,?,?,1,?],f32>
}
