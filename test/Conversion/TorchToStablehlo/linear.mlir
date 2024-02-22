// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.mm$basic$static(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[3,3],f32> -> tensor<3x3xf32>
// CHECK:         %[[T2:.*]] = stablehlo.dot %[[T0]], %[[T1]] : (tensor<2x3xf32>, tensor<3x3xf32>) -> tensor<2x3xf32>
// CHECK:         %[[T3:.*]] = tensor.cast %[[T2]] : tensor<2x3xf32> to tensor<2x3xf32>
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
// CHECK:         %[[T2:.*]] = stablehlo.dot %[[T0]], %[[T1]] : (tensor<?x3xf32>, tensor<3x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = tensor.cast %[[T2]] : tensor<?x?xf32> to tensor<?x?xf32>
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
// CHECK:         %[[T9:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T1]], %[[T8]], dims = [0, 1, 2] : (tensor<10x4x5xf32>, tensor<3xi64>) -> tensor<10x4x5xf32>
// CHECK:         %[[T10:.*]] = stablehlo.dot_general %[[T0]], %[[T9]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<10x3x4xf32>, tensor<10x4x5xf32>) -> tensor<10x3x5xf32>
// CHECK:         %[[T11:.*]] = tensor.cast %[[T10]] : tensor<10x3x5xf32> to tensor<10x3x5xf32>
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
// CHECK:         %[[T9:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T1]], %[[T8]], dims = [0, 1, 2] : (tensor<?x4x?xf32>, tensor<3xi64>) -> tensor<?x4x?xf32>
// CHECK:         %[[T10:.*]] = stablehlo.dot_general %[[T0]], %[[T9]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<?x?x4xf32>, tensor<?x4x?xf32>) -> tensor<?x?x?xf32>
// CHECK:         %[[T11:.*]] = tensor.cast %[[T10]] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
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
// CHECK:         %[[T9:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T0]], %[[T8]], dims = [1, 2] : (tensor<256x120xf32>, tensor<3xi64>) -> tensor<4x256x120xf32>
// CHECK:         %[[T10:.*]] = stablehlo.dot_general %[[T9]], %[[T1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x256x120xf32>, tensor<4x120x256xf32>) -> tensor<4x256x256xf32>
// CHECK:         %[[T11:.*]] = tensor.cast %[[T10]] : tensor<4x256x256xf32> to tensor<4x256x256xf32>
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
// CHECK:         %[[T9:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T1]], %[[T8]], dims = [1, 2] : (tensor<256x?xf32>, tensor<3xi64>) -> tensor<4x256x?xf32>
// CHECK:         %[[T10:.*]] = stablehlo.dot_general %[[T0]], %[[T9]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x?x256xf32>, tensor<4x256x?xf32>) -> tensor<4x?x?xf32>
// CHECK:         %[[T11:.*]] = tensor.cast %[[T10]] : tensor<4x?x?xf32> to tensor<4x?x?xf32>
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
// CHECK:         %[[T7:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T1]], %[[T6]], dims = [1] : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:         %[[T8:.*]] = stablehlo.dot_general %[[T0]], %[[T7]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x?x256xf32>, tensor<1x256xf32>) -> tensor<1x?xf32>
// CHECK:         %[[T9:.*]] = tensor.cast %[[T8]] : tensor<1x?xf32> to tensor<1x?xf32>
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
// CHECK:         %[[T7:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T0]], %[[T6]], dims = [1] : (tensor<256xf32>, tensor<2xi64>) -> tensor<?x256xf32>
// CHECK:         %[[T8:.*]] = stablehlo.dot_general %[[T7]], %[[T1]], batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<?x256xf32>, tensor<?x256x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T9:.*]] = tensor.cast %[[T8]] : tensor<?x?xf32> to tensor<?x?xf32>
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
// CHECK:         %[[T2:.*]] = stablehlo.dot %[[T0]], %[[T1]] : (tensor<?x256xf32>, tensor<256xf32>) -> tensor<?xf32>
// CHECK:         %[[T3:.*]] = tensor.cast %[[T2]] : tensor<?xf32> to tensor<?xf32>
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
// CHECK:         %[[T2:.*]] = stablehlo.dot %[[T0]], %[[T1]] : (tensor<256xf32>, tensor<256x?xf32>) -> tensor<?xf32>
// CHECK:         %[[T3:.*]] = tensor.cast %[[T2]] : tensor<?xf32> to tensor<?xf32>
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
// CHECK:         %[[T2:.*]] = stablehlo.dot %[[T0]], %[[T1]] : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = tensor.cast %[[T2]] : tensor<f32> to tensor<f32>
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
// CHECK:         %[[T1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<256x256xf32>
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
// CHECK:         %[[T9:.*]] = stablehlo.dynamic_broadcast_in_dim %[[T1]], %[[T8]], dims = [1, 2] : (tensor<256x256xf32>, tensor<3xi64>) -> tensor<?x256x256xf32>
// CHECK:         %[[T10:.*]] = stablehlo.dot_general %[[T0]], %[[T9]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<?x?x256xf32>, tensor<?x256x256xf32>) -> tensor<?x?x256xf32>
// CHECK:         %[[T11:.*]] = tensor.cast %[[T10]] : tensor<?x?x256xf32> to tensor<?x?x256xf32>
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
// CHECK:         %[[T1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<256x256xf32>
// CHECK:         %[[T2:.*]] = stablehlo.dot %[[T0]], %[[T1]] : (tensor<?x256xf32>, tensor<256x256xf32>) -> tensor<?x256xf32>
// CHECK:         %[[T3:.*]] = tensor.cast %[[T2]] : tensor<?x256xf32> to tensor<?x256xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?x256xf32> -> !torch.vtensor<[?,256],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?,256],f32>
func.func @torch.aten.mm$proj(%arg0: !torch.vtensor<[?,256],f32>) -> !torch.vtensor<[?,256],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256xf32>) : !torch.vtensor<[256,256],f32>
  %1 = torch.aten.mm %arg0, %0 : !torch.vtensor<[?,256],f32>, !torch.vtensor<[256,256],f32> -> !torch.vtensor<[?,256],f32>
  return %1 : !torch.vtensor<[?,256],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>,
// CHECK-SAME:                                 %[[ARG_1:.*]]: !torch.vtensor<[?,?,3,3],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?,3,3],f32> -> tensor<?x?x3x3xf32>
// CHECK:           %[[T_2:.*]] = torch.constant.none
// CHECK:           %[[T_4:.*]] = torch.constant.int 2
// CHECK:           %[[T_5:.*]] = torch.constant.int 1
// CHECK:           %[[T_6:.*]] = torch.constant.int 4
// CHECK:           %[[T_7:.*]] = torch.constant.int 3
// CHECK:           %[[T_8:.*]] = torch_c.to_i64 %[[T_7]]
// CHECK:           %[[T_9:.*]] = torch.prim.ListConstruct %[[T_4]], %[[T_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_10:.*]] = torch.prim.ListConstruct %[[T_6]], %[[T_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_11:.*]] = torch.prim.ListConstruct %[[T_7]], %[[T_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[T_13:.*]] = torch.constant.bool false
// CHECK:           %[[T_14:.*]] = stablehlo.convolution(%[[T_0]], %[[T_1]])
// CHECK-SAME{LITERAL}:                                                 dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[4, 4], [2, 2]], rhs_dilate = [3, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_15:.*]] = torch_c.from_builtin_tensor %[[T_14]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[T_15]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.convolution(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,3,3],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %false = torch.constant.bool false
  %5 = torch.aten.convolution %arg0, %arg1, %none, %1, %2, %3, %false, %4, %int3 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %5 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$bias(
// CHECK-SAME:                                           %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, %[[ARG_1:.*]]: !torch.vtensor<[?,?,3,3],f32>,
// CHECK-SAME:                                           %[[ARG_2:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?,3,3],f32> -> tensor<?x?x3x3xf32>
// CHECK:           %[[T_2:.*]] = torch_c.to_builtin_tensor %[[ARG_2]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int4 = torch.constant.int 4
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %[[T_3:.*]] = torch_c.to_i64 %int3
// CHECK:           %[[T_4:.*]] = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_5:.*]] = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_6:.*]] = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_7:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %false = torch.constant.bool false
// CHECK:           %[[T_8:.*]] = stablehlo.convolution(%[[T_0]], %[[T_1]])
// CHECK{LITERAL}:                                                     dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 1], pad = [[4, 4], [2, 2]], rhs_dilate = [3, 1]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[T_9:.*]] = tensor.dim %[[T_2]], %[[IDX_0]] : tensor<?xf32>
// CHECK:           %[[T_10:.*]] = arith.index_cast %[[T_9]] : index to i64
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[T_11:.*]] = tensor.from_elements %[[T_10]], %[[VAL_0]], %[[VAL_0]] : tensor<3xi64>
// CHECK:           %[[T_12:.*]] = stablehlo.dynamic_reshape %[[T_2]], %[[T_11]] : (tensor<?xf32>, tensor<3xi64>) -> tensor<?x1x1xf32>
// CHECK:           %[[T_13:.*]] = chlo.broadcast_add %[[T_8]], %[[T_12]] : (tensor<?x?x?x?xf32>, tensor<?x1x1xf32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_14:.*]] = torch_c.from_builtin_tensor %[[T_13]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[T_14]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.convolution$bias(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,3,3],f32>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %false = torch.constant.bool false
  %5 = torch.aten.convolution %arg0, %arg1, %arg2, %1, %2, %3, %false, %4, %int3 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,3,3],f32>, !torch.vtensor<[?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %5 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$transposed_basic(
// CHECK-SAME:                                                       %[[ARG_0:.*]]: !torch.vtensor<[1,2,7,7],f32>,
// CHECK-SAME:                                                       %[[ARG_1:.*]]: !torch.vtensor<[2,4,3,3],f32>) -> !torch.vtensor<[1,4,9,9],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[1,2,7,7],f32> -> tensor<1x2x7x7xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[2,4,3,3],f32> -> tensor<2x4x3x3xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %none = torch.constant.none
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[T_2:.*]] = torch_c.to_i64 %int1
// CHECK:           %[[T_3:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_4:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_5:.*]] = stablehlo.transpose %[[T_1]], dims = [2, 3, 1, 0] : (tensor<2x4x3x3xf32>) -> tensor<3x3x4x2xf32>
// CHECK:           %[[T_6:.*]] = stablehlo.reverse %[[T_5]], dims = [0, 1] : tensor<3x3x4x2xf32>
// CHECK:           %[[T_7:.*]] = stablehlo.convolution(%[[T_0]], %[[T_6]])
// CHECK{LITERAL}:                   dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<3x3x4x2xf32>) -> tensor<1x4x9x9xf32>
// CHECK:           %[[T_8:.*]] = torch_c.from_builtin_tensor %[[T_7]] : tensor<1x4x9x9xf32> -> !torch.vtensor<[1,4,9,9],f32>
// CHECK:           return %[[T_8]] : !torch.vtensor<[1,4,9,9],f32>
func.func @torch.aten.convolution$transposed_basic(%arg0: !torch.vtensor<[1,2,7,7],f32>, %arg1: !torch.vtensor<[2,4,3,3],f32>) -> !torch.vtensor<[1,4,9,9],f32> {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.convolution %arg0, %arg1, %none, %1, %0, %1, %true, %0, %int1 : !torch.vtensor<[1,2,7,7],f32>, !torch.vtensor<[2,4,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,9,9],f32>
  return %2 : !torch.vtensor<[1,4,9,9],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$transposed_stride(
// CHECK-SAME:                                                        %[[ARG_0:.*]]: !torch.vtensor<[1,2,7,7],f32>,
// CHECK-SAME:                                                        %[[ARG_1:.*]]: !torch.vtensor<[2,4,3,3],f32>) -> !torch.vtensor<[1,4,15,15],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[1,2,7,7],f32> -> tensor<1x2x7x7xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[2,4,3,3],f32> -> tensor<2x4x3x3xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %none = torch.constant.none
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[T_2:.*]] = torch_c.to_i64 %int1
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[T_3:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_4:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_5:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_6:.*]] = stablehlo.transpose %[[T_1]], dims = [2, 3, 1, 0] : (tensor<2x4x3x3xf32>) -> tensor<3x3x4x2xf32>
// CHECK:           %[[T_7:.*]] = stablehlo.reverse %[[T_6]], dims = [0, 1] : tensor<3x3x4x2xf32>
// CHECK:           %[[T_8:.*]] = stablehlo.convolution(%[[T_0]], %[[T_7]])
// CHECK{LITERAL}:         dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<3x3x4x2xf32>) -> tensor<1x4x15x15xf32>
// CHECK:           %[[T_9:.*]] = torch_c.from_builtin_tensor %[[T_8]] : tensor<1x4x15x15xf32> -> !torch.vtensor<[1,4,15,15],f32>
// CHECK:           return %[[T_9]] : !torch.vtensor<[1,4,15,15],f32>
func.func @torch.aten.convolution$transposed_stride(%arg0: !torch.vtensor<[1,2,7,7],f32>, %arg1: !torch.vtensor<[2,4,3,3],f32>) -> !torch.vtensor<[1,4,15,15],f32> {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.convolution %arg0, %arg1, %none, %2, %0, %1, %true, %0, %int1 : !torch.vtensor<[1,2,7,7],f32>, !torch.vtensor<[2,4,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,15,15],f32>
  return %3 : !torch.vtensor<[1,4,15,15],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$transposed_outputpadding(
// CHECK-SAME:                                                               %[[ARG_0:.*]]: !torch.vtensor<[1,2,7,7],f32>,
// CHECK-SAME:                                                               %[[ARG_1:.*]]: !torch.vtensor<[2,4,3,3],f32>) -> !torch.vtensor<[1,4,16,16],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[1,2,7,7],f32> -> tensor<1x2x7x7xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[2,4,3,3],f32> -> tensor<2x4x3x3xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %none = torch.constant.none
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[T_2:.*]] = torch_c.to_i64 %int1
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[T_3:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_4:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_5:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_6:.*]] = stablehlo.transpose %[[T_1]], dims = [2, 3, 1, 0] : (tensor<2x4x3x3xf32>) -> tensor<3x3x4x2xf32>
// CHECK:           %[[T_7:.*]] = stablehlo.reverse %[[T_6]], dims = [0, 1] : tensor<3x3x4x2xf32>
// CHECK:           %[[T_8:.*]] = stablehlo.convolution(%[[T_0]], %[[T_7]])
// CHECK{LITERAL}:           dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 3], [2, 3]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x2x7x7xf32>, tensor<3x3x4x2xf32>) -> tensor<1x4x16x16xf32>
// CHECK:           %[[T_9:.*]] = torch_c.from_builtin_tensor %[[T_8:.*]] : tensor<1x4x16x16xf32> -> !torch.vtensor<[1,4,16,16],f32>
// CHECK:           return %[[T_9]] : !torch.vtensor<[1,4,16,16],f32>
func.func @torch.aten.convolution$transposed_outputpadding(%arg0: !torch.vtensor<[1,2,7,7],f32>, %arg1: !torch.vtensor<[2,4,3,3],f32>) -> !torch.vtensor<[1,4,16,16],f32> {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.convolution %arg0, %arg1, %none, %2, %0, %1, %true, %1, %int1 : !torch.vtensor<[1,2,7,7],f32>, !torch.vtensor<[2,4,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,16,16],f32>
  return %3 : !torch.vtensor<[1,4,16,16],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.convolution$transposed_groups(
// CHECK-SAME:                                                        %[[ARG_0:.*]]: !torch.vtensor<[1,2,7,7],f32>,
// CHECK-SAME:                                                        %[[ARG_1:.*]]: !torch.vtensor<[2,2,3,3],f32>) -> !torch.vtensor<[1,4,15,15],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[1,2,7,7],f32> -> tensor<1x2x7x7xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[2,2,3,3],f32> -> tensor<2x2x3x3xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %none = torch.constant.none
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %[[T_2:.*]] = torch_c.to_i64 %int2
// CHECK:           %[[T_3:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_4:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_5:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_6:.*]] = stablehlo.transpose %[[T_1]], dims = [2, 3, 1, 0] : (tensor<2x2x3x3xf32>) -> tensor<3x3x2x2xf32>
// CHECK:           %[[T_7:.*]] = stablehlo.reverse %6, dims = [0, 1] : tensor<3x3x2x2xf32>
// CHECK:           %c0 = arith.constant 0 : index
// CHECK:           %dim = tensor.dim %[[T_7]], %c0 : tensor<3x3x2x2xf32>
// CHECK:           %[[T_8:.*]] = arith.index_cast %dim : index to i64
// CHECK:           %c1 = arith.constant 1 : index
// CHECK:           %dim_0 = tensor.dim %[[T_7]], %c1 : tensor<3x3x2x2xf32>
// CHECK:           %[[T_9:.*]] = arith.index_cast %dim_0 : index to i64
// CHECK:           %c2 = arith.constant 2 : index
// CHECK:           %dim_1 = tensor.dim %[[T_7]], %c2 : tensor<3x3x2x2xf32>
// CHECK:           %[[T_10:.*]] = arith.index_cast %dim_1 : index to i64
// CHECK:           %c3 = arith.constant 3 : index
// CHECK:           %dim_2 = tensor.dim %[[T_7]], %c3 : tensor<3x3x2x2xf32>
// CHECK:           %[[T_11:.*]] = arith.index_cast %dim_2 : index to i64
// CHECK:           %c2_i64 = arith.constant 2 : i64
// CHECK:           %[[T_12:.*]] = arith.divsi %[[T_11]], %c2_i64 : i64
// CHECK:           %[[T_13:.*]] = arith.muli %[[T_10]], %c2_i64 : i64
// CHECK:           %from_elements = tensor.from_elements %[[T_8]], %[[T_9]], %[[T_10]], %c2_i64, %[[T_12]] : tensor<5xi64>
// CHECK:           %[[T_14:.*]] = stablehlo.dynamic_reshape %[[T_7]], %from_elements : (tensor<3x3x2x2xf32>, tensor<5xi64>) -> tensor<3x3x2x2x1xf32>
// CHECK:           %[[T_15:.*]] = stablehlo.transpose %[[T_14]], dims = [0, 1, 3, 2, 4] : (tensor<3x3x2x2x1xf32>) -> tensor<3x3x2x2x1xf32>
// CHECK:           %from_elements_3 = tensor.from_elements %[[T_8]], %[[T_9]], %[[T_13]], %[[T_12]] : tensor<4xi64>
// CHECK:           %[[T_16:.*]] = stablehlo.dynamic_reshape %[[T_15]], %from_elements_3 : (tensor<3x3x2x2x1xf32>, tensor<4xi64>) -> tensor<3x3x4x1xf32>
// CHECK:           %[[T_17:.*]] = stablehlo.convolution(%[[T_0]], %[[T_16]])
// CHECK{LITERAL}:          dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 2 : i64} : (tensor<1x2x7x7xf32>, tensor<3x3x4x1xf32>) -> tensor<1x4x15x15xf32>
// CHECK:           %[[T_18:.*]] = torch_c.from_builtin_tensor %[[T_17]] : tensor<1x4x15x15xf32> -> !torch.vtensor<[1,4,15,15],f32>
// CHECK:           return %[[T_18]] : !torch.vtensor<[1,4,15,15],f32>
func.func @torch.aten.convolution$transposed_groups(%arg0: !torch.vtensor<[1,2,7,7],f32>, %arg1: !torch.vtensor<[2,2,3,3],f32>) -> !torch.vtensor<[1,4,15,15],f32> {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.convolution %arg0, %arg1, %none, %2, %0, %1, %true, %0, %int2 : !torch.vtensor<[1,2,7,7],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,15,15],f32>
  return %3 : !torch.vtensor<[1,4,15,15],f32>
}
