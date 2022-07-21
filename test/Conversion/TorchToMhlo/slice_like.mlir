// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.slice.strided$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch_c.to_i64 %[[INT0]]
// CHECK:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT2]]
// CHECK:         %[[INT9223372036854775807:.*]] = torch.constant.int 9223372036854775807
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT9223372036854775807]]
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T6:.*]] = arith.subi %[[C0_I64]], %[[T5]] : i64
// CHECK:         %[[T7:.*]] = arith.maxsi %[[T6]], %[[T1]] : i64
// CHECK:         %[[T8:.*]] = arith.minsi %[[T5]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.addi %[[T5]], %[[T8]] : i64
// CHECK:         %[[T10:.*]] = arith.cmpi sge, %[[T8]], %[[C0_I64]] : i64
// CHECK:         %[[T11:.*]] = arith.select %[[T10]], %[[T8]], %[[T9]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T12:.*]] = arith.subi %[[C0_I64_0]], %[[T5]] : i64
// CHECK:         %[[T13:.*]] = arith.maxsi %[[T12]], %[[T3]] : i64
// CHECK:         %[[T14:.*]] = arith.minsi %[[T5]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.addi %[[T5]], %[[T14]] : i64
// CHECK:         %[[T16:.*]] = arith.cmpi sge, %[[T14]], %[[C0_I64_0]] : i64
// CHECK:         %[[T17:.*]] = arith.select %[[T16]], %[[T14]], %[[T15]] : i64
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         %[[T18:.*]] = tensor.dim %[[T0]], %[[C0_1]] : tensor<?x?x?xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T20:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T22:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T23:.*]] = arith.index_cast %[[T22]] : index to i64
// CHECK:         %[[C0_I64_2:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T24:.*]] = arith.cmpi eq, %[[T17]], %[[C0_I64_2]] : i64
// CHECK:         %[[T25:.*]] = arith.select %[[T24]], %[[T19]], %[[T17]] : i64
// CHECK:         %[[T26:.*]] = tensor.from_elements %[[T11]], %[[C0_I64_2]], %[[C0_I64_2]] : tensor<3xi64>
// CHECK:         %[[T27:.*]] = tensor.from_elements %[[T25]], %[[T21]], %[[T23]] : tensor<3xi64>
// CHECK:         %[[T28:.*]] = tensor.from_elements %[[T2]], %[[C1_I64]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T29:.*]] = "mhlo.real_dynamic_slice"(%[[T0]], %[[T26]], %[[T27]], %[[T28]]) : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert %[[T29]] : tensor<?x?x?xf32>
// CHECK:         %[[T31:.*]] = torch_c.from_builtin_tensor %[[T30]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[T31]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.slice.strided$slice_like(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int9223372036854775807, %int2 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.slice.strided.static$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[2,65,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch_c.to_i64 %[[INT0]]
// CHECK:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT2]]
// CHECK:         %[[INT9223372036854775807:.*]] = torch.constant.int 9223372036854775807
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT9223372036854775807]]
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x65x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T6:.*]] = arith.subi %[[C0_I64]], %[[T5]] : i64
// CHECK:         %[[T7:.*]] = arith.maxsi %[[T6]], %[[T1]] : i64
// CHECK:         %[[T8:.*]] = arith.minsi %[[T5]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.addi %[[T5]], %[[T8]] : i64
// CHECK:         %[[T10:.*]] = arith.cmpi sge, %[[T8]], %[[C0_I64]] : i64
// CHECK:         %[[T11:.*]] = arith.select %[[T10]], %[[T8]], %[[T9]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T12:.*]] = arith.subi %[[C0_I64_0]], %[[T5]] : i64
// CHECK:         %[[T13:.*]] = arith.maxsi %[[T12]], %[[T3]] : i64
// CHECK:         %[[T14:.*]] = arith.minsi %[[T5]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.addi %[[T5]], %[[T14]] : i64
// CHECK:         %[[T16:.*]] = arith.cmpi sge, %[[T14]], %[[C0_I64_0]] : i64
// CHECK:         %[[T17:.*]] = arith.select %[[T16]], %[[T14]], %[[T15]] : i64
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         %[[T18:.*]] = tensor.dim %[[T0]], %[[C0_1]] : tensor<4x65x256xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T20:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x65x256xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T22:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x65x256xf32>
// CHECK:         %[[T23:.*]] = arith.index_cast %[[T22]] : index to i64
// CHECK:         %[[C0_I64_2:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T24:.*]] = arith.cmpi eq, %[[T17]], %[[C0_I64_2]] : i64
// CHECK:         %[[T25:.*]] = arith.select %[[T24]], %[[T19]], %[[T17]] : i64
// CHECK:         %[[T26:.*]] = tensor.from_elements %[[T11]], %[[C0_I64_2]], %[[C0_I64_2]] : tensor<3xi64>
// CHECK:         %[[T27:.*]] = tensor.from_elements %[[T25]], %[[T21]], %[[T23]] : tensor<3xi64>
// CHECK:         %[[T28:.*]] = tensor.from_elements %[[T2]], %[[C1_I64]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T29:.*]] = "mhlo.real_dynamic_slice"(%[[T0]], %[[T26]], %[[T27]], %[[T28]]) : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x65x256xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert(%[[T29]]) : (tensor<?x65x256xf32>) -> tensor<2x65x256xf32>
// CHECK:         %[[T31:.*]] = torch_c.from_builtin_tensor %[[T30]] : tensor<2x65x256xf32> -> !torch.vtensor<[2,65,256],f32>
// CHECK:         return %[[T31]] : !torch.vtensor<[2,65,256],f32>
func.func @torch.aten.slice.strided.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[2,65,256],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int9223372036854775807, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,65,256],f32>
  return %0 : !torch.vtensor<[2,65,256],f32>
}


// CHECK-LABEL:  func.func @torch.aten.slice.last$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch_c.to_i64 %[[INT0]]
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT1]]
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T6:.*]] = arith.subi %[[C0_I64]], %[[T5]] : i64
// CHECK:         %[[T7:.*]] = arith.maxsi %[[T6]], %[[T3]] : i64
// CHECK:         %[[T8:.*]] = arith.minsi %[[T5]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.addi %[[T5]], %[[T8]] : i64
// CHECK:         %[[T10:.*]] = arith.cmpi sge, %[[T8]], %[[C0_I64]] : i64
// CHECK:         %[[T11:.*]] = arith.select %[[T10]], %[[T8]], %[[T9]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T12:.*]] = arith.subi %[[C0_I64_0]], %[[T5]] : i64
// CHECK:         %[[T13:.*]] = arith.maxsi %[[T12]], %[[T1]] : i64
// CHECK:         %[[T14:.*]] = arith.minsi %[[T5]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.addi %[[T5]], %[[T14]] : i64
// CHECK:         %[[T16:.*]] = arith.cmpi sge, %[[T14]], %[[C0_I64_0]] : i64
// CHECK:         %[[T17:.*]] = arith.select %[[T16]], %[[T14]], %[[T15]] : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T18:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i64
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         %[[T20:.*]] = tensor.dim %[[T0]], %[[C1_1]] : tensor<?x?x?xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T22:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T23:.*]] = arith.index_cast %[[T22]] : index to i64
// CHECK:         %[[C0_I64_2:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T24:.*]] = arith.cmpi eq, %[[T17]], %[[C0_I64_2]] : i64
// CHECK:         %[[T25:.*]] = arith.select %[[T24]], %[[T21]], %[[T17]] : i64
// CHECK:         %[[T26:.*]] = tensor.from_elements %[[C0_I64_2]], %[[T11]], %[[C0_I64_2]] : tensor<3xi64>
// CHECK:         %[[T27:.*]] = tensor.from_elements %[[T19]], %[[T25]], %[[T23]] : tensor<3xi64>
// CHECK:         %[[T28:.*]] = tensor.from_elements %[[C1_I64]], %[[T2]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T29:.*]] = "mhlo.real_dynamic_slice"(%[[T0]], %[[T26]], %[[T27]], %[[T28]]) : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert(%[[T29]]) : (tensor<?x?x?xf32>) -> tensor<?x1x?xf32>
// CHECK:         %[[T31:.*]] = torch_c.from_builtin_tensor %[[T30]] : tensor<?x1x?xf32> -> !torch.vtensor<[?,1,?],f32>
// CHECK:         return %[[T31]] : !torch.vtensor<[?,1,?],f32>
func.func @torch.aten.slice.last$slice_like(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,1,?],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int-1, %int0, %int1 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,1,?],f32>
  return %0 : !torch.vtensor<[?,1,?],f32>
}


// CHECK-LABEL:  func.func @torch.aten.slice.last.static$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,1,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch_c.to_i64 %[[INT0]]
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INT1]]
// CHECK:         %[[INT:.*]]-1 = torch.constant.int -1
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT]]-1
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x65x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[T6:.*]] = arith.subi %[[C0_I64]], %[[T5]] : i64
// CHECK:         %[[T7:.*]] = arith.maxsi %[[T6]], %[[T3]] : i64
// CHECK:         %[[T8:.*]] = arith.minsi %[[T5]], %[[T7]] : i64
// CHECK:         %[[T9:.*]] = arith.addi %[[T5]], %[[T8]] : i64
// CHECK:         %[[T10:.*]] = arith.cmpi sge, %[[T8]], %[[C0_I64]] : i64
// CHECK:         %[[T11:.*]] = arith.select %[[T10]], %[[T8]], %[[T9]] : i64
// CHECK:         %[[C0_I64_0:.*]] = arith.constant 0 : i64
// CHECK:         %[[T12:.*]] = arith.subi %[[C0_I64_0]], %[[T5]] : i64
// CHECK:         %[[T13:.*]] = arith.maxsi %[[T12]], %[[T1]] : i64
// CHECK:         %[[T14:.*]] = arith.minsi %[[T5]], %[[T13]] : i64
// CHECK:         %[[T15:.*]] = arith.addi %[[T5]], %[[T14]] : i64
// CHECK:         %[[T16:.*]] = arith.cmpi sge, %[[T14]], %[[C0_I64_0]] : i64
// CHECK:         %[[T17:.*]] = arith.select %[[T16]], %[[T14]], %[[T15]] : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T18:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x65x256xf32>
// CHECK:         %[[T19:.*]] = arith.index_cast %[[T18]] : index to i64
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         %[[T20:.*]] = tensor.dim %[[T0]], %[[C1_1]] : tensor<4x65x256xf32>
// CHECK:         %[[T21:.*]] = arith.index_cast %[[T20]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T22:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x65x256xf32>
// CHECK:         %[[T23:.*]] = arith.index_cast %[[T22]] : index to i64
// CHECK:         %[[C0_I64_2:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T24:.*]] = arith.cmpi eq, %[[T17]], %[[C0_I64_2]] : i64
// CHECK:         %[[T25:.*]] = arith.select %[[T24]], %[[T21]], %[[T17]] : i64
// CHECK:         %[[T26:.*]] = tensor.from_elements %[[C0_I64_2]], %[[T11]], %[[C0_I64_2]] : tensor<3xi64>
// CHECK:         %[[T27:.*]] = tensor.from_elements %[[T19]], %[[T25]], %[[T23]] : tensor<3xi64>
// CHECK:         %[[T28:.*]] = tensor.from_elements %[[C1_I64]], %[[T2]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T29:.*]] = "mhlo.real_dynamic_slice"(%[[T0]], %[[T26]], %[[T27]], %[[T28]]) : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x?x256xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert(%[[T29]]) : (tensor<4x?x256xf32>) -> tensor<4x1x256xf32>
// CHECK:         %[[T31:.*]] = torch_c.from_builtin_tensor %[[T30]] : tensor<4x1x256xf32> -> !torch.vtensor<[4,1,256],f32>
// CHECK:         return %[[T31]] : !torch.vtensor<[4,1,256],f32>
func.func @torch.aten.slice.last.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,1,256],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int-1, %int0, %int1 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,1,256],f32>
  return %0 : !torch.vtensor<[4,1,256],f32>
}


// CHECK-LABEL:  func.func @torch.aten.slice.none$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:         %[[T1:.*]] = torch_c.to_i64 %[[INT2]]
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T0]], %[[C1_0]] : tensor<?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T8:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i64
// CHECK:         %[[C0_I64_1:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T10:.*]] = arith.cmpi eq, %[[T3]], %[[C0_I64_1]] : i64
// CHECK:         %[[T11:.*]] = arith.select %[[T10]], %[[T7]], %[[T3]] : i64
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[C0_I64_1]], %[[C0_I64]], %[[C0_I64_1]] : tensor<3xi64>
// CHECK:         %[[T13:.*]] = tensor.from_elements %[[T5]], %[[T11]], %[[T9]] : tensor<3xi64>
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[C1_I64]], %[[T1]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T15:.*]] = "mhlo.real_dynamic_slice"(%[[T0]], %[[T12]], %[[T13]], %[[T14]]) : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:         %[[T16:.*]] = mhlo.convert %[[T15]] : tensor<?x?x?xf32>
// CHECK:         %[[T17:.*]] = torch_c.from_builtin_tensor %[[T16]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[T17]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.slice.none$slice_like(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.aten.slice.Tensor %arg0, %int1, %none, %none, %int2 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.slice.none.static$slice_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,33,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,65,256],f32> -> tensor<4x65x256xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:         %[[T1:.*]] = torch_c.to_i64 %[[INT2]]
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x65x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x65x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T0]], %[[C1_0]] : tensor<4x65x256xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T8:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x65x256xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i64
// CHECK:         %[[C0_I64_1:.*]] = arith.constant 0 : i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T10:.*]] = arith.cmpi eq, %[[T3]], %[[C0_I64_1]] : i64
// CHECK:         %[[T11:.*]] = arith.select %[[T10]], %[[T7]], %[[T3]] : i64
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[C0_I64_1]], %[[C0_I64]], %[[C0_I64_1]] : tensor<3xi64>
// CHECK:         %[[T13:.*]] = tensor.from_elements %[[T5]], %[[T11]], %[[T9]] : tensor<3xi64>
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[C1_I64]], %[[T1]], %[[C1_I64]] : tensor<3xi64>
// CHECK:         %[[T15:.*]] = "mhlo.real_dynamic_slice"(%[[T0]], %[[T12]], %[[T13]], %[[T14]]) : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x?x256xf32>
// CHECK:         %[[T16:.*]] = mhlo.convert(%[[T15]]) : (tensor<4x?x256xf32>) -> tensor<4x33x256xf32>
// CHECK:         %[[T17:.*]] = torch_c.from_builtin_tensor %[[T16]] : tensor<4x33x256xf32> -> !torch.vtensor<[4,33,256],f32>
// CHECK:         return %[[T17]] : !torch.vtensor<[4,33,256],f32>
func.func @torch.aten.slice.none.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,33,256],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.aten.slice.Tensor %arg0, %int1, %none, %none, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,33,256],f32>
  return %0 : !torch.vtensor<[4,33,256],f32>
}
