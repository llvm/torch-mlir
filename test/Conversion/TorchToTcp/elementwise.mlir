// RUN: torch-mlir-opt <%s -convert-torch-to-tcp -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.addtensor$samerank(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = tcp.add %[[T0]], %[[T1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addtensor$samerank(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.addtensor$diffrank(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:         %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[TO_BUILTIN0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[TO_BUILTIN1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:         %[[EXPAND_SHAPE:.*]] = tensor.expand_shape %[[TO_BUILTIN1]]
// CHECK-SAME:                           [0, 1]
// CHECK-SAME:                           : tensor<?xf32> into tensor<1x?xf32>
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[TO_BUILTIN0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[BROADCAST:.*]] = tcp.broadcast %[[EXPAND_SHAPE]], %[[DIM0]] {axes = [0]} : tensor<1x?xf32>, index -> tensor<?x?xf32>
// CHECK:         %[[ADD:.*]] = tcp.add %[[TO_BUILTIN0]], %[[BROADCAST]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[FROM_BUILTIN:.*]] = torch_c.from_builtin_tensor %[[ADD]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[FROM_BUILTIN]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addtensor$diffrank(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.addtensor$rank0(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:         %[[ARG1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[TO_BUILTIN0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[TO_BUILTIN1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:         %[[EXPAND_SHAPE:.*]] = tensor.expand_shape %[[TO_BUILTIN1]]
// CHECK-SAME:                           []
// CHECK-SAME:                           : tensor<f32> into tensor<1x1xf32>
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[TO_BUILTIN0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[TO_BUILTIN0]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[BROADCAST:.*]] = tcp.broadcast %[[EXPAND_SHAPE]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:         %[[ADD:.*]] = tcp.add %[[TO_BUILTIN0]], %[[BROADCAST]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         %[[FROM_BUILTIN:.*]] = torch_c.from_builtin_tensor %[[ADD]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[FROM_BUILTIN]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addtensor$rank0(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}
