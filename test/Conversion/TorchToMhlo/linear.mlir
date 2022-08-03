// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.mm$basic$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[3,3],f32> -> tensor<3x3xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot"(%[[T0]], %[[T1]]) : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<2x3xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[2,3],f32>
func.func @torch.aten.mm$basic$static(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.mm$basic$dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,3],f32> -> tensor<?x3xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[3,?],f32> -> tensor<3x?xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot"(%[[T0]], %[[T1]]) : (tensor<?x3xf32>, tensor<3x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?x?xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.mm$basic$dynamic(%arg0: !torch.vtensor<[?,3],f32>, %arg1: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[3,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.bmm$basic$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[10,3,4],f32>, %[[ARG1:.*]]: !torch.vtensor<[10,4,5],f32>) -> !torch.vtensor<[10,3,5],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[10,3,4],f32> -> tensor<10x3x4xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[10,4,5],f32> -> tensor<10x4x5xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<10x4x5xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<10x4x5xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T1]], %[[C2]] : tensor<10x4x5xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T7]] : tensor<3xi64>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[T8]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<10x4x5xf32>, tensor<3xi64>) -> tensor<10x4x5xf32>
// CHECK:         %[[T10:.*]] = "mhlo.dot_general"(%[[T0]], %[[T9]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<10x3x4xf32>, tensor<10x4x5xf32>) -> tensor<10x3x5xf32>
// CHECK:         %[[T11:.*]] = mhlo.convert %[[T10]] : tensor<10x3x5xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<10x3x5xf32> -> !torch.vtensor<[10,3,5],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[10,3,5],f32>
func.func @torch.aten.bmm$basic$static(%arg0: !torch.vtensor<[10,3,4],f32>, %arg1: !torch.vtensor<[10,4,5],f32>) -> !torch.vtensor<[10,3,5],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,3,4],f32>, !torch.vtensor<[10,4,5],f32> -> !torch.vtensor<[10,3,5],f32>
  return %0 : !torch.vtensor<[10,3,5],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.bmm$basic$dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,4],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,4,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,4],f32> -> tensor<?x?x4xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,4,?],f32> -> tensor<?x4x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x4x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<?x4x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T1]], %[[C2]] : tensor<?x4x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T7]] : tensor<3xi64>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[T8]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x4x?xf32>, tensor<3xi64>) -> tensor<?x4x?xf32>
// CHECK:         %[[T10:.*]] = "mhlo.dot_general"(%[[T0]], %[[T9]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x4xf32>, tensor<?x4x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T11:.*]] = mhlo.convert %[[T10]] : tensor<?x?x?xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.bmm$basic$dynamic(%arg0: !torch.vtensor<[?,?,4],f32>, %arg1: !torch.vtensor<[?,4,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,4],f32>, !torch.vtensor<[?,4,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

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

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$basic$dynamic(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[4,?,256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,?,256],f32> -> tensor<4x?x256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256,?],f32> -> tensor<256x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<4x?x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T1]], %[[C0_0]] : tensor<256x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<256x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T7]] : tensor<3xi64>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[T8]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<256x?xf32>, tensor<3xi64>) -> tensor<4x256x?xf32>
// CHECK:         %[[T10:.*]] = "mhlo.dot_general"(%[[T0]], %[[T9]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x?x256xf32>, tensor<4x256x?xf32>) -> tensor<4x?x?xf32>
// CHECK:         %[[T11:.*]] = mhlo.convert %[[T10]] : tensor<4x?x?xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<4x?x?xf32> -> !torch.vtensor<[4,?,?],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[4,?,?],f32>
func.func @torch.aten.matmul$basic$dynamic(%arg0: !torch.vtensor<[4,?,256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[4,?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[4,?,?],f32>
  return %0 : !torch.vtensor<[4,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$3dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,?,256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256],f32>) -> !torch.vtensor<[1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,?,256],f32> -> tensor<1x?x256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<1x?x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T1]], %[[C0_0]] : tensor<256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T3]], %[[T5]] : tensor<2xi64>
// CHECK:         %[[T7:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[T6]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:         %[[T8:.*]] = "mhlo.dot_general"(%[[T0]], %[[T7]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x?x256xf32>, tensor<1x256xf32>) -> tensor<1x?xf32>
// CHECK:         %[[T9:.*]] = mhlo.convert %[[T8]] : tensor<1x?xf32>
// CHECK:         %[[T10:.*]] = torch_c.from_builtin_tensor %[[T9]] : tensor<1x?xf32> -> !torch.vtensor<[1,?],f32>
// CHECK:         return %[[T10]] : !torch.vtensor<[1,?],f32>
func.func @torch.aten.matmul$3dx1d(%arg0: !torch.vtensor<[1,?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[1,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[1,?],f32>
  return %0 : !torch.vtensor<[1,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$1dx3d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,256,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,256,?],f32> -> tensor<?x256x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<?x256x?xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T0]], %[[C0_0]] : tensor<256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[T6:.*]] = tensor.from_elements %[[T3]], %[[T5]] : tensor<2xi64>
// CHECK:         %[[T7:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T0]], %[[T6]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>, tensor<2xi64>) -> tensor<?x256xf32>
// CHECK:         %[[T8:.*]] = "mhlo.dot_general"(%[[T7]], %[[T1]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<?x256xf32>, tensor<?x256x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = mhlo.convert %[[T8]] : tensor<?x?xf32>
// CHECK:         %[[T10:.*]] = torch_c.from_builtin_tensor %[[T9]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T10]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.matmul$1dx3d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[?,256,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[?,256,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$2dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,256],f32> -> tensor<?x256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot"(%[[T0]], %[[T1]]) : (tensor<?x256xf32>, tensor<256xf32>) -> tensor<?xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?],f32>
func.func @torch.aten.matmul$2dx1d(%arg0: !torch.vtensor<[?,256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$1dx2d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256,?],f32> -> tensor<256x?xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot"(%[[T0]], %[[T1]]) : (tensor<256xf32>, tensor<256x?xf32>) -> tensor<?xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?],f32>
func.func @torch.aten.matmul$1dx2d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256,?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256,?],f32> -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$1dx1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[256],f32>, %[[ARG1:.*]]: !torch.vtensor<[256],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[256],f32> -> tensor<256xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot"(%[[T0]], %[[T1]]) : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<f32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[],f32>
func.func @torch.aten.matmul$1dx1d(%arg0: !torch.vtensor<[256],f32>, %arg1: !torch.vtensor<[256],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[256],f32>, !torch.vtensor<[256],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.matmul$proj(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,256],f32>) -> !torch.vtensor<[?,?,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,256],f32> -> tensor<?x?x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[T2:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x256xf32>
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : index to i64
// CHECK:         %[[C0_0:.*]] = arith.constant 0 : index
// CHECK:         %[[T4:.*]] = tensor.dim %[[T1]], %[[C0_0]] : tensor<256x256xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[T6:.*]] = tensor.dim %[[T1]], %[[C1]] : tensor<256x256xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[T6]] : index to i64
// CHECK:         %[[T8:.*]] = tensor.from_elements %[[T3]], %[[T5]], %[[T7]] : tensor<3xi64>
// CHECK:         %[[T9:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[T1]], %[[T8]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<256x256xf32>, tensor<3xi64>) -> tensor<?x256x256xf32>
// CHECK:         %[[T10:.*]] = "mhlo.dot_general"(%[[T0]], %[[T9]]) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x256xf32>, tensor<?x256x256xf32>) -> tensor<?x?x256xf32>
// CHECK:         %[[T11:.*]] = mhlo.convert %[[T10]] : tensor<?x?x256xf32>
// CHECK:         %[[T12:.*]] = torch_c.from_builtin_tensor %[[T11]] : tensor<?x?x256xf32> -> !torch.vtensor<[?,?,256],f32>
// CHECK:         return %[[T12]] : !torch.vtensor<[?,?,256],f32>
func.func @torch.aten.matmul$proj(%arg0: !torch.vtensor<[?,?,256],f32>) -> !torch.vtensor<[?,?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[?,?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,?,256],f32>
  return %1 : !torch.vtensor<[?,?,256],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.mm$proj(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,256],f32> -> tensor<?x256xf32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[T2:.*]] = "mhlo.dot"(%[[T0]], %[[T1]]) : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         %[[T3:.*]] = mhlo.convert %[[T2]] : tensor<?x256xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?x256xf32> -> !torch.vtensor<[?,256],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?,256],f32>
func.func @torch.aten.mm$proj(%arg0: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,256],f32>
  return %1 : !torch.vtensor<[?,256],f32>
}

