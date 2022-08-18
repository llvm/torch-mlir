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
// CHECK:         %[[T29:.*]] = mhlo.real_dynamic_slice %[[T0]], %[[T26]], %[[T27]], %[[T28]] : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
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
// CHECK:         %[[T29:.*]] = mhlo.real_dynamic_slice %[[T0]], %[[T26]], %[[T27]], %[[T28]] : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x65x256xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert %[[T29]] : tensor<2x65x256xf32>
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
// CHECK:         %[[T29:.*]] = mhlo.real_dynamic_slice %[[T0]], %[[T26]], %[[T27]], %[[T28]] : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x1x?xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert %[[T29]] : tensor<?x1x?xf32>
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
// CHECK:         %[[T29:.*]] = mhlo.real_dynamic_slice %[[T0]], %[[T26]], %[[T27]], %[[T28]] : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x1x256xf32>
// CHECK:         %[[T30:.*]] = mhlo.convert %[[T29]] : tensor<4x1x256xf32>
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
// CHECK:         %[[T15:.*]] = mhlo.real_dynamic_slice %[[T0]], %[[T12]], %[[T13]], %[[T14]] : (tensor<?x?x?xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x?x?xf32>
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
// CHECK:         %[[T15:.*]] = mhlo.real_dynamic_slice %[[T0]], %[[T12]], %[[T13]], %[[T14]] : (tensor<4x65x256xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x33x256xf32>
// CHECK:         %[[T16:.*]] = mhlo.convert %[[T15]] : tensor<4x33x256xf32>
// CHECK:         %[[T17:.*]] = torch_c.from_builtin_tensor %[[T16]] : tensor<4x33x256xf32> -> !torch.vtensor<[4,33,256],f32>
// CHECK:         return %[[T17]] : !torch.vtensor<[4,33,256],f32>
func.func @torch.aten.slice.none.static$slice_like(%arg0: !torch.vtensor<[4,65,256],f32>) -> !torch.vtensor<[4,33,256],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none
  %0 = torch.aten.slice.Tensor %arg0, %int1, %none, %none, %int2 : !torch.vtensor<[4,65,256],f32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,33,256],f32>
  return %0 : !torch.vtensor<[4,33,256],f32>
}

// CHECK-LABEL:  func.func @torch.aten.view$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INTneg1:.*]] = torch.constant.int -1
// CHECK:         %[[INT224:.*]] = torch.constant.int 224
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INTneg1]], %[[INT224]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INTneg1]]
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT224]]
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T4:.*]] = arith.muli %[[C1_I64]], %[[T2]] : i64
// CHECK:         %[[T5:.*]] = arith.muli %[[T4]], %[[T3]] : i64
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : i64 to index
// CHECK:         %[[T7:.*]] = tensor.from_elements %[[T2]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T8:.*]] = mhlo.compute_reshape_shape %[[T6]], %[[T7]] : index, tensor<2xi64> -> tensor<2xi64>
// CHECK:         %[[T9:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T8]] : (tensor<?x?x?x?xf32>, tensor<2xi64>) -> tensor<?x224xf32>
// CHECK:         %[[T10:.*]] = torch_c.from_builtin_tensor %[[T9]] : tensor<?x224xf32> -> !torch.vtensor<[?,224],f32>
// CHECK:         return %[[T10]] : !torch.vtensor<[?,224],f32>
func.func @torch.aten.view$basic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,224],f32> {
  %int-1 = torch.constant.int -1
  %int224 = torch.constant.int 224
  %0 = torch.prim.ListConstruct %int-1, %int224 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,224],f32>
  return %1 : !torch.vtensor<[?,224],f32>
}

// CHECK-LABEL:  func.func @torch.aten.reshape$basic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?,?],f32> -> tensor<?x?x?x?x?xf32>
// CHECK:         %[[INTneg1:.*]] = torch.constant.int -1
// CHECK:         %[[INT120:.*]] = torch.constant.int 120
// CHECK:         %[[INT4:.*]] = torch.constant.int 4
// CHECK:         %[[INT64:.*]] = torch.constant.int 64
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INTneg1]], %[[INT120]], %[[INT4]], %[[INT64]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %[[INTneg1]]
// CHECK:         %[[T3:.*]] = torch_c.to_i64 %[[INT120]]
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %[[INT4]]
// CHECK:         %[[T5:.*]] = torch_c.to_i64 %[[INT64]]
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T6:.*]] = arith.muli %[[C1_I64]], %[[T2]] : i64
// CHECK:         %[[T7:.*]] = arith.muli %[[T6]], %[[T3]] : i64
// CHECK:         %[[T8:.*]] = arith.muli %[[T7]], %[[T4]] : i64
// CHECK:         %[[T9:.*]] = arith.muli %[[T8]], %[[T5]] : i64
// CHECK:         %[[T10:.*]] = arith.index_cast %[[T9]] : i64 to index
// CHECK:         %[[T11:.*]] = tensor.from_elements %[[T2]], %[[T3]], %[[T4]], %[[T5]] : tensor<4xi64>
// CHECK:         %[[T12:.*]] = mhlo.compute_reshape_shape %[[T10]], %[[T11]] : index, tensor<4xi64> -> tensor<4xi64>
// CHECK:         %[[T13:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T12]] : (tensor<?x?x?x?x?xf32>, tensor<4xi64>) -> tensor<?x120x4x64xf32>
// CHECK:         %[[T14:.*]] = torch_c.from_builtin_tensor %[[T13]] : tensor<?x120x4x64xf32> -> !torch.vtensor<[?,120,4,64],f32>
// CHECK:         return %[[T14]] : !torch.vtensor<[?,120,4,64],f32>
func.func @torch.aten.reshape$basic(%arg0: !torch.vtensor<[?,?,?,?,?],f32>) -> !torch.vtensor<[?,120,4,64],f32> {
  %int-1 = torch.constant.int -1
  %int120 = torch.constant.int 120
  %int4 = torch.constant.int 4
  %int64 = torch.constant.int 64
  %0 = torch.prim.ListConstruct %int-1, %int120, %int4, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reshape %arg0, %0 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,120,4,64],f32>
  return %1 : !torch.vtensor<[?,120,4,64],f32>
}

// CHECK-LABEL:  func.func @torch.aten.view$minus1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,3,?,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,3,?,?],f32> -> tensor<2x3x?x?xf32>
// CHECK:         %[[INTneg1:.*]] = torch.constant.int -1
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T1:.*]] = torch.aten.size.int %[[ARG0]], %[[INT0]] : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
// CHECK:         %[[T2:.*]] = torch.aten.size.int %[[ARG0]], %[[INT1]] : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
// CHECK:         %[[T3:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]], %[[INTneg1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %[[T1]]
// CHECK:         %[[T5:.*]] = torch_c.to_i64 %[[T2]]
// CHECK:         %[[T6:.*]] = torch_c.to_i64 %[[INTneg1]]
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T7:.*]] = arith.muli %[[C1_I64]], %[[T4]] : i64
// CHECK:         %[[T8:.*]] = arith.muli %[[T7]], %[[T5]] : i64
// CHECK:         %[[T9:.*]] = arith.muli %[[T8]], %[[T6]] : i64
// CHECK:         %[[T10:.*]] = arith.index_cast %[[T9]] : i64 to index
// CHECK:         %[[T11:.*]] = tensor.from_elements %[[T4]], %[[T5]], %[[T6]] : tensor<3xi64>
// CHECK:         %[[T12:.*]] = mhlo.compute_reshape_shape %[[T10]], %[[T11]] : index, tensor<3xi64> -> tensor<3xi64>
// CHECK:         %[[T13:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T12]] : (tensor<2x3x?x?xf32>, tensor<3xi64>) -> tensor<2x3x?xf32>
// CHECK:         %[[T14:.*]] = torch_c.from_builtin_tensor %[[T13]] : tensor<2x3x?xf32> -> !torch.vtensor<[2,3,?],f32>
// CHECK:         return %[[T14]] : !torch.vtensor<[2,3,?],f32>
func.func @torch.aten.view$minus1(%arg0: !torch.vtensor<[2,3,?,?],f32>) -> !torch.vtensor<[2,3,?],f32> {
  %int-1 = torch.constant.int -1
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
  %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[2,3,?,?],f32>, !torch.int -> !torch.int
  %2 = torch.prim.ListConstruct %0, %1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[2,3,?,?],f32>, !torch.list<int> -> !torch.vtensor<[2,3,?],f32>
  return %3 : !torch.vtensor<[2,3,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.view$to_rank1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT1]] : (!torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = mhlo.reshape %[[T0]] : (tensor<f32>) -> tensor<1xf32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[1],f32>
func.func @torch.aten.view$to_rank1(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1],f32>
  return %1 : !torch.vtensor<[1],f32>
}
// CHECK-LABEL:  func.func @torch.aten.view$to_rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1],f32> -> tensor<1xf32>
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:         %[[T2:.*]] = mhlo.reshape %[[T0]] : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[],f32>
func.func @torch.aten.view$to_rank0(%arg0: !torch.vtensor<[1],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}
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

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim$1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,?,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,1,?,1,?],f32> -> tensor<?x1x?x1x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T5:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T0]], %[[C4]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T2]], %[[T4]], %[[T6]], %[[T8]] : tensor<4xi64>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T9]] : (tensor<?x1x?x1x?xf32>, tensor<4xi64>) -> tensor<?x?x1x?xf32>
// CHECK:         %[[T11:.*]] = torch_c.from_builtin_tensor %[[T10]] : tensor<?x?x1x?xf32> -> !torch.vtensor<[?,?,1,?],f32>
// CHECK:         return %[[T11]] : !torch.vtensor<[?,?,1,?],f32>
func.func @torch.aten.squeeze.dim$1(%arg0: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,?,1,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.squeeze.dim %arg0, %int1 : !torch.vtensor<[?,1,?,1,?],f32>, !torch.int -> !torch.vtensor<[?,?,1,?],f32>
  return %0 : !torch.vtensor<[?,?,1,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.squeeze.dim$from_end(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,1,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,1,?,1,?],f32> -> tensor<?x1x?x1x?xf32>
// CHECK:         %[[INT:.*]]-2 = torch.constant.int -2
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T5:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T0]], %[[C4]] : tensor<?x1x?x1x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T2]], %[[T4]], %[[T6]], %[[T8]] : tensor<4xi64>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T9]] : (tensor<?x1x?x1x?xf32>, tensor<4xi64>) -> tensor<?x1x?x?xf32>
// CHECK:         %[[T11:.*]] = torch_c.from_builtin_tensor %[[T10]] : tensor<?x1x?x?xf32> -> !torch.vtensor<[?,1,?,?],f32>
// CHECK:         return %[[T11]] : !torch.vtensor<[?,1,?,?],f32>
func.func @torch.aten.squeeze.dim$from_end(%arg0: !torch.vtensor<[?,1,?,1,?],f32>) -> !torch.vtensor<[?,1,?,?],f32> {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.squeeze.dim %arg0, %int-2 : !torch.vtensor<[?,1,?,1,?],f32>, !torch.int -> !torch.vtensor<[?,1,?,?],f32>
  return %0 : !torch.vtensor<[?,1,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.squeeze$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,2,2],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,1,2,1,2],f32> -> tensor<2x1x2x1x2xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<2x1x2x1x2xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<2x1x2x1x2xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[T5:.*]] = tensor.dim %[[T0]], %[[C4]] : tensor<2x1x2x1x2xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[T7:.*]] = tensor.from_elements %[[T2]], %[[T4]], %[[T6]] : tensor<3xi64>
// CHECK:         %[[T8:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T7]] : (tensor<2x1x2x1x2xf32>, tensor<3xi64>) -> tensor<2x2x2xf32>
// CHECK:         %[[T9:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<2x2x2xf32> -> !torch.vtensor<[2,2,2],f32>
// CHECK:         return %[[T9]] : !torch.vtensor<[2,2,2],f32>
func.func @torch.aten.squeeze$static(%arg0: !torch.vtensor<[2,1,2,1,2],f32>) -> !torch.vtensor<[2,2,2],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[2,1,2,1,2],f32> -> !torch.vtensor<[2,2,2],f32>
  return %0 : !torch.vtensor<[2,2,2],f32>
}

// CHECK-LABEL:  func.func @torch.aten.unsqueeze$dim$0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1,?,?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T5:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[C1_I64]], %[[T2]], %[[T4]], %[[T6]], %[[T8]] : tensor<5xi64>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T9]] : (tensor<?x?x?x?xf32>, tensor<5xi64>) -> tensor<1x?x?x?x?xf32>
// CHECK:         %[[T11:.*]] = torch_c.from_builtin_tensor %[[T10]] : tensor<1x?x?x?x?xf32> -> !torch.vtensor<[1,?,?,?,?],f32>
// CHECK:         return %[[T11]] : !torch.vtensor<[1,?,?,?,?],f32>
func.func @torch.aten.unsqueeze$dim$0(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[1,?,?,?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[1,?,?,?,?],f32>
  return %0 : !torch.vtensor<[1,?,?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.unsqueeze$dim$1(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,1,?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T5:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T2]], %[[C1_I64]], %[[T4]], %[[T6]], %[[T8]] : tensor<5xi64>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T9]] : (tensor<?x?x?x?xf32>, tensor<5xi64>) -> tensor<?x1x?x?x?xf32>
// CHECK:         %[[T11:.*]] = torch_c.from_builtin_tensor %[[T10]] : tensor<?x1x?x?x?xf32> -> !torch.vtensor<[?,1,?,?,?],f32>
// CHECK:         return %[[T11]] : !torch.vtensor<[?,1,?,?,?],f32>
func.func @torch.aten.unsqueeze$dim$1(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,1,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.unsqueeze %arg0, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,1,?,?,?],f32>
  return %0 : !torch.vtensor<[?,1,?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.unsqueeze$from_end(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[INT:.*]]-2 = torch.constant.int -2
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T1:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[T1]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T3:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[T3]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T5:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[T5]] : index to i64
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T9:.*]] = tensor.from_elements %[[T2]], %[[T4]], %[[T6]], %[[C1_I64]], %[[T8]] : tensor<5xi64>
// CHECK:         %[[T10:.*]] = mhlo.dynamic_reshape %[[T0]], %[[T9]] : (tensor<?x?x?x?xf32>, tensor<5xi64>) -> tensor<?x?x?x1x?xf32>
// CHECK:         %[[T11:.*]] = torch_c.from_builtin_tensor %[[T10]] : tensor<?x?x?x1x?xf32> -> !torch.vtensor<[?,?,?,1,?],f32>
// CHECK:         return %[[T11]] : !torch.vtensor<[?,?,?,1,?],f32>
func.func @torch.aten.unsqueeze$from_end(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,1,?],f32> {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.unsqueeze %arg0, %int-2 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?,?,1,?],f32>
  return %0 : !torch.vtensor<[?,?,?,1,?],f32>
}
