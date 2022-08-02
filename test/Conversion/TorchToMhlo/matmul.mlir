// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.mm$basic$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[3,3],f32> -> tensor<3x3xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<2x3xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[2,3],f32>
func.func @torch.aten.mm$basic$static(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL:  func.func @torch.aten.mm$basic$dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,3],f32> -> tensor<?x3xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[3,?],f32> -> tensor<3x?xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x3xf32>, tensor<3x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.mm$basic$dynamic(%arg0: !torch.vtensor<[?,3],f32>, %arg1: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[3,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.bmm$basic$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[10,3,4],f32>, %[[ARG1:.*]]: !torch.vtensor<[10,4,5],f32>) -> !torch.vtensor<[10,3,5],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[10,3,4],f32> -> tensor<10x3x4xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[10,4,5],f32> -> tensor<10x4x5xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<10x3x4xf32>, tensor<10x4x5xf32>) -> tensor<10x3x5xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<10x3x5xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<10x3x5xf32> -> !torch.vtensor<[10,3,5],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[10,3,5],f32>
func.func @torch.aten.bmm$basic$static(%arg0: !torch.vtensor<[10,3,4],f32>, %arg1: !torch.vtensor<[10,4,5],f32>) -> !torch.vtensor<[10,3,5],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,3,4],f32>, !torch.vtensor<[10,4,5],f32> -> !torch.vtensor<[10,3,5],f32>
  return %0 : !torch.vtensor<[10,3,5],f32>
}

// CHECK-LABEL:  func.func @torch.aten.bmm$basic$dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,4],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,4,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,4],f32> -> tensor<?x?x4xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,4,?],f32> -> tensor<?x4x?xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x4xf32>, tensor<?x4x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.bmm$basic$dynamic(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[?,4,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,4],f32>, !torch.vtensor<[?,4,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$basic$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256,120],f32>, %[[ARG1:.*]]: !torch.vtensor<[4,120,256],f32>) -> !torch.vtensor<[4,256,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256,120],f32> -> tensor<256x120xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[4,120,256],f32> -> tensor<4x120x256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<4x120x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C0_0]] : tensor<256x120xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<256x120xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T7]] : tensor<3xi64>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[T8]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<256x120xf32>, tensor<3xi64>) -> tensor<4x256x120xf32>
// CHECK:         %[[T10:.*]] = "mhlo.dot_general"(%[[T9]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x256x120xf32>, tensor<4x120x256xf32>) -> tensor<4x256x256xf32>
// CHECK:         %[[T11:.*]] = mhlo.convert %[[T10]] : tensor<4x256x256xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<4x256x256xf32> -> !torch.vtensor<[4,256,256],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[4,256,256],f32>
func.func @torch.aten.matmul$basic$static(%arg0: !torch.vtensor<[256,120],f32>, %arg1: !torch.vtensor<[4,120,256],f32>) -> !torch.vtensor<[4,256,256],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256,120],f32>, !torch.vtensor<[4,120,256],f32> -> !torch.vtensor<[4,256,256],f32>
  return %0 : !torch.vtensor<[4,256,256],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$basic$dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,?,256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,?,256],f32> -> tensor<4x?x256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256,?],f32> -> tensor<256x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x?x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<4x?x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<4x?x256xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T8:.*]] = arith.muli %[[C1_I64]], %[[T3]] : i64
// CHECK:         %[[T9:.*]] = arith.muli %[[T8]], %[[T5]] : i64
// CHECK:         %[[T10:.*]] = tensor.from_elements %[[T9]], %[[T7]] : tensor<2xi64>
// CHECK:         %[[T11:.*]] = "mhlo.dynamic_reshape"(%[[T0]], %[[T10]]) : (tensor<4x?x256xf32>, tensor<2xi64>) -> tensor<?x256xf32>
// CHECK:         %[[T12:.*]] = "mhlo.dot_general"(%[[T11]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T13:.*]] = tensor.dim %[[T12]], %[[C0_0]] : tensor<?x?xf32>
// CHECK:         %[[T14:.*]] = arith.index_cast %[[T13]] : index to i64
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         %[[T15:.*]] = tensor.dim %[[T12]], %[[C1_1]] : tensor<?x?xf32>
// CHECK:         %[[T16:.*]] = arith.index_cast %[[T15]] : index to i64
// CHECK:         %[[T17:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T16]] : tensor<3xi64>
// CHECK:         %[[T18:.*]] = "mhlo.dynamic_reshape"(%[[T12]], %[[T17]]) : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CHECK:         %[[T19:.*]] = mhlo.convert(%[[T18]]) : (tensor<?x?x?xf32>) -> tensor<4x?x?xf32>
// CHECK:         %[[T20:.*]] = torch_c.from_builtin_tensor %[[T19]] : tensor<4x?x?xf32> -> !torch.vtensor<[4,?,?],f32>
// CHECK:         return %[[T20]] : !torch.vtensor<[4,?,?],f32>
func.func @torch.aten.matmul$basic$dynamic(%arg0: !torch.vtensor<[4,?,256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[4,?,?],f32>
  return %0 : !torch.vtensor<[4,?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$3dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,?,256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256],f32>) -> !torch.vtensor<[1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,?,256],f32> -> tensor<1x?x256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[T3]], %[[C1_I64]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T1]], %[[T4]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<256x1xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T0]], %[[C0_0]] : tensor<1x?x256xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T8:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<1x?x256xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T10:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<1x?x256xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i64
// CHECK:         %[[C1_I64_1:.*]] = arith.constant 1 : i64
// CHECK:         %[[T12:.*]] = arith.muli %[[C1_I64_1]], %[[T7]] : i64
// CHECK:         %[[T13:.*]] = arith.muli %[[T12]], %[[T9]] : i64
// CHECK:         %[[T14:.*]] = tensor.from_elements %[[T13]], %[[T11]] : tensor<2xi64>
// CHECK:         %[[T15:.*]] = "mhlo.dynamic_reshape"(%[[T0]], %[[T14]]) : (tensor<1x?x256xf32>, tensor<2xi64>) -> tensor<?x256xf32>
// CHECK:         %[[T16:.*]] = "mhlo.dot_general"(%[[T15]], %[[T5]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x1xf32>) -> tensor<?x1xf32>
// CHECK:         %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:         %[[T17:.*]] = tensor.dim %[[T16]], %[[C0_2]] : tensor<?x1xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[T17]] : index to i64
// CHECK:         %[[C1_3:.*]] = arith.constant 1 : index
// CHECK:         %[[T19:.*]] = tensor.dim %[[T16]], %[[C1_3]] : tensor<?x1xf32>
// CHECK:         %[[T20:.*]] = arith.index_cast %[[T19]] : index to i64
// CHECK:         %[[T21:.*]] = tensor.from_elements %[[T7]], %[[T9]], %[[T20]] : tensor<3xi64>
// CHECK:         %[[T22:.*]] = "mhlo.dynamic_reshape"(%[[T16]], %[[T21]]) : (tensor<?x1xf32>, tensor<3xi64>) -> tensor<?x?x1xf32>
// CHECK:         %[[C0_4:.*]] = arith.constant 0 : index
// CHECK:         %[[T23:.*]] = tensor.dim %[[T22]], %[[C0_4]] : tensor<?x?x1xf32>
// CHECK:         %[[T24:.*]] = arith.index_cast %[[T23]] : index to i64
// CHECK:         %[[C1_5:.*]] = arith.constant 1 : index
// CHECK:         %[[T25:.*]] = tensor.dim %[[T22]], %[[C1_5]] : tensor<?x?x1xf32>
// CHECK:         %[[T26:.*]] = arith.index_cast %[[T25]] : index to i64
// CHECK:         %[[C2_6:.*]] = arith.constant 2 : index
// CHECK:         %[[T27:.*]] = tensor.dim %[[T22]], %[[C2_6]] : tensor<?x?x1xf32>
// CHECK:         %[[T28:.*]] = arith.index_cast %[[T27]] : index to i64
// CHECK:         %[[C1_I64_7:.*]] = arith.constant 1 : i64
// CHECK:         %[[T29:.*]] = arith.muli %[[C1_I64_7]], %[[T26]] : i64
// CHECK:         %[[T30:.*]] = arith.muli %[[T29]], %[[T28]] : i64
// CHECK:         %[[T31:.*]] = tensor.from_elements %[[T24]], %[[T30]] : tensor<2xi64>
// CHECK:         %[[T32:.*]] = "mhlo.dynamic_reshape"(%[[T22]], %[[T31]]) : (tensor<?x?x1xf32>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:         %[[T33:.*]] = mhlo.convert(%[[T32]]) : (tensor<?x?xf32>) -> tensor<1x?xf32>
// CHECK:         %[[T34:.*]] = torch_c.from_builtin_tensor %[[T33]] : tensor<1x?xf32> -> !torch.vtensor<[1,?],f32>
// CHECK:         return %[[T34]] : !torch.vtensor<[1,?],f32>
func.func @torch.aten.matmul$3dx1d(%arg0: !torch.vtensor<[1,?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[1,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[1,?],f32>
  return %0 : !torch.vtensor<[1,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$1dx3d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,256,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,256,?],f32> -> tensor<?x256x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T0]], %[[T4]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T1]], %[[C0_0]] : tensor<?x256x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:         %[[T8:.*]] = tensor.dim %[[T5]], %[[C0_1]] : tensor<1x256xf32>
// CHECK:         %[[T9:.*]] = arith.index_cast %[[T8]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T10:.*]] = tensor.dim %[[T5]], %[[C1]] : tensor<1x256xf32>
// CHECK:         %[[T11:.*]] = arith.index_cast %[[T10]] : index to i64
// CHECK:         %[[T12:.*]] = tensor.from_elements %[[T7]], %[[T9]], %[[T11]] : tensor<3xi64>
// CHECK:         %[[T13:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T5]], %[[T12]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x256xf32>, tensor<3xi64>) -> tensor<?x1x256xf32>
// CHECK:         %[[T14:.*]] = "mhlo.dot_general"(%[[T13]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x1x256xf32>, tensor<?x256x?xf32>) -> tensor<?x1x?xf32>
// CHECK:         %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:         %[[T15:.*]] = tensor.dim %[[T14]], %[[C0_2]] : tensor<?x1x?xf32>
// CHECK:         %[[T16:.*]] = arith.index_cast %[[T15]] : index to i64
// CHECK:         %[[C1_3:.*]] = arith.constant 1 : index
// CHECK:         %[[T17:.*]] = tensor.dim %[[T14]], %[[C1_3]] : tensor<?x1x?xf32>
// CHECK:         %[[T18:.*]] = arith.index_cast %[[T17]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T19:.*]] = tensor.dim %[[T14]], %[[C2]] : tensor<?x1x?xf32>
// CHECK:         %[[T20:.*]] = arith.index_cast %[[T19]] : index to i64
// CHECK:         %[[C1_I64_4:.*]] = arith.constant 1 : i64
// CHECK:         %[[T21:.*]] = arith.muli %[[C1_I64_4]], %[[T18]] : i64
// CHECK:         %[[T22:.*]] = arith.muli %[[T21]], %[[T20]] : i64
// CHECK:         %[[T23:.*]] = tensor.from_elements %[[T16]], %[[T22]] : tensor<2xi64>
// CHECK:         %[[T24:.*]] = "mhlo.dynamic_reshape"(%[[T14]], %[[T23]]) : (tensor<?x1x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
// CHECK:         %[[T25:.*]] = mhlo.convert %[[T24]] : tensor<?x?xf32>
// CHECK:         %[[T26:.*]] = torch_c.from_builtin_tensor %[[T25]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T26]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.matmul$1dx3d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[?,256,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[?,256,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$2dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,256],f32> -> tensor<?x256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[T3]], %[[C1_I64]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T1]], %[[T4]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<256x1xf32>
// CHECK:         %[[T6:.*]] = "mhlo.dot_general"(%[[T0]], %[[T5]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x1xf32>) -> tensor<?x1xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T6]], %[[C0_0]] : tensor<?x1xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T9:.*]] = tensor.dim %[[T6]], %[[C1]] : tensor<?x1xf32>
// CHECK:         %[[T10:.*]] = arith.index_cast %[[T9]] : index to i64
// CHECK:         %[[C1_I64_1:.*]] = arith.constant 1 : i64
// CHECK:         %[[T11:.*]] = arith.muli %[[C1_I64_1]], %[[T8]] : i64
// CHECK:         %[[T12:.*]] = arith.muli %[[T11]], %[[T10]] : i64
// CHECK:         %[[T13:.*]] = tensor.from_elements %[[T12]] : tensor<1xi64>
// CHECK:         %[[T14:.*]] = "mhlo.dynamic_reshape"(%[[T6]], %[[T13]]) : (tensor<?x1xf32>, tensor<1xi64>) -> tensor<?xf32>
// CHECK:         %[[T15:.*]] = mhlo.convert %[[T14]] : tensor<?xf32>
// CHECK:         %[[T16:.*]] = torch_c.from_builtin_tensor %[[T15]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:         return %[[T16]] : !torch.vtensor<[?],f32>
func.func @torch.aten.matmul$2dx1d(%arg0: !torch.vtensor<[?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$1dx2d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256,?],f32> -> tensor<256x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T0]], %[[T4]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:         %[[T6:.*]] = "mhlo.dot_general"(%[[T5]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x?xf32>) -> tensor<1x?xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T7:.*]] = tensor.dim %[[T6]], %[[C0_0]] : tensor<1x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[T7]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T9:.*]] = tensor.dim %[[T6]], %[[C1]] : tensor<1x?xf32>
// CHECK:         %[[T10:.*]] = arith.index_cast %[[T9]] : index to i64
// CHECK:         %[[C1_I64_1:.*]] = arith.constant 1 : i64
// CHECK:         %[[T11:.*]] = arith.muli %[[C1_I64_1]], %[[T8]] : i64
// CHECK:         %[[T12:.*]] = arith.muli %[[T11]], %[[T10]] : i64
// CHECK:         %[[T13:.*]] = tensor.from_elements %[[T12]] : tensor<1xi64>
// CHECK:         %[[T14:.*]] = "mhlo.dynamic_reshape"(%[[T6]], %[[T13]]) : (tensor<1x?xf32>, tensor<1xi64>) -> tensor<?xf32>
// CHECK:         %[[T15:.*]] = mhlo.convert %[[T14]] : tensor<?xf32>
// CHECK:         %[[T16:.*]] = torch_c.from_builtin_tensor %[[T15]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:         return %[[T16]] : !torch.vtensor<[?],f32>
func.func @torch.aten.matmul$1dx2d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$1dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T4:.*]] = tensor.from_elements %[[C1_I64]], %[[T3]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_reshape"(%[[T0]], %[[T4]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T1]], %[[C0_0]] : tensor<256xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C1_I64_1:.*]] = arith.constant 1 : i64
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T7]], %[[C1_I64_1]] : tensor<2xi64>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_reshape"(%[[T1]], %[[T8]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<256x1xf32>
// CHECK:         %[[T10:.*]] = "mhlo.dot_general"(%[[T5]], %[[T9]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
// CHECK:         %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:         %[[T11:.*]] = tensor.dim %[[T10]], %[[C0_2]] : tensor<1x1xf32>
// CHECK:         %[[T12:.*]] = arith.index_cast %[[T11]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T13:.*]] = tensor.dim %[[T10]], %[[C1]] : tensor<1x1xf32>
// CHECK:         %[[T14:.*]] = arith.index_cast %[[T13]] : index to i64
// CHECK:         %[[C1_I64_3:.*]] = arith.constant 1 : i64
// CHECK:         %[[T15:.*]] = arith.muli %[[C1_I64_3]], %[[T12]] : i64
// CHECK:         %[[T16:.*]] = arith.muli %[[T15]], %[[T14]] : i64
// CHECK:         %[[T17:.*]] = tensor.from_elements %[[T16]] : tensor<1xi64>
// CHECK:         %[[T18:.*]] = "mhlo.dynamic_reshape"(%[[T10]], %[[T17]]) : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:         %[[T19:.*]] = "mhlo.reshape"(%[[T18]]) : (tensor<1xf32>) -> tensor<f32>
// CHECK:         %[[T20:.*]] = mhlo.convert %[[T19]] : tensor<f32>
// CHECK:         %[[T21:.*]] = torch_c.from_builtin_tensor %[[T20]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T21]] : !torch.vtensor<[],f32>
func.func @torch.aten.matmul$1dx1d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:  func.func @torch.aten.matmul$proj(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,256],f32>) -> !torch.vtensor<[?,?,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,256],f32> -> tensor<?x?x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x256xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[T8:.*]] = arith.muli %[[C1_I64]], %[[T3]] : i64
// CHECK:         %[[T9:.*]] = arith.muli %[[T8]], %[[T5]] : i64
// CHECK:         %[[T10:.*]] = tensor.from_elements %[[T9]], %[[T7]] : tensor<2xi64>
// CHECK:         %[[T11:.*]] = "mhlo.dynamic_reshape"(%[[T0]], %[[T10]]) : (tensor<?x?x256xf32>, tensor<2xi64>) -> tensor<?x256xf32>
// CHECK:         %[[T12:.*]] = "mhlo.dot_general"(%[[T11]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T13:.*]] = tensor.dim %[[T12]], %[[C0_0]] : tensor<?x256xf32>
// CHECK:         %[[T14:.*]] = arith.index_cast %[[T13]] : index to i64
// CHECK:         %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:         %[[T15:.*]] = tensor.dim %[[T12]], %[[C1_1]] : tensor<?x256xf32>
// CHECK:         %[[T16:.*]] = arith.index_cast %[[T15]] : index to i64
// CHECK:         %[[T17:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T16]] : tensor<3xi64>
// CHECK:         %[[T18:.*]] = "mhlo.dynamic_reshape"(%[[T12]], %[[T17]]) : (tensor<?x256xf32>, tensor<3xi64>) -> tensor<?x?x256xf32>
// CHECK:         %[[T19:.*]] = mhlo.convert %[[T18]] : tensor<?x?x256xf32>
// CHECK:         %[[T20:.*]] = torch_c.from_builtin_tensor %[[T19]] : tensor<?x?x256xf32> -> !torch.vtensor<[?,?,256],f32>
// CHECK:         return %[[T20]] : !torch.vtensor<[?,?,256],f32>
func.func @torch.aten.matmul$proj(%arg0: !torch.vtensor<[?,?,256],f32>) -> !torch.vtensor<[?,?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[?,?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,?,256],f32>
  return %1 : !torch.vtensor<[?,?,256],f32>
}

// CHECK-LABEL:  func.func @torch.aten.mm$proj(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,256],f32> -> tensor<?x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot_general"(%[[T0]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?x256xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?x256xf32> -> !torch.vtensor<[?,256],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?,256],f32>
func.func @torch.aten.mm$proj(%arg0: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,256],f32>
  return %1 : !torch.vtensor<[?,256],f32>
}
