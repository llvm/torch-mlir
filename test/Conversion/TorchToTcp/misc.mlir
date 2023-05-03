// RUN: torch-mlir-opt <%s -convert-torch-to-tcp -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>} : tensor<4xf32>
// CHECK:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:         return %[[T2]] : !torch.vtensor<[4],f32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],f32> {
  %0 = torch.vtensor.literal(dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],si32> {
//       CHECK:  %[[T1:.+]] = tcp.const {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : tensor<4xi32>
//       CHECK:  %[[T2:.+]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xi32> -> !torch.vtensor<[4],si32>
//       CHECK:  return %[[T2]] : !torch.vtensor<[4],si32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],si32> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi32>) : !torch.vtensor<[4],si32>
  return %0 : !torch.vtensor<[4],si32>
}

// -----

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],ui8> {
//       CHECK:  %[[T1:.+]] = tcp.const {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
//       CHECK:  %[[T2:.+]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xi8> -> !torch.vtensor<[4],ui8>
//       CHECK:  return %[[T2]] : !torch.vtensor<[4],ui8>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],ui8> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  return %0 : !torch.vtensor<[4],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_f32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<0.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] : tensor<f32> into tensor<1x1xf32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.zeros_f32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.zeros %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_si32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],si32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i32> into tensor<1x1xi32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.zeros_si32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],si32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.zeros %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],si32>
  return %1 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_ui8(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],ui8> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i8>} : tensor<i8>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i8> into tensor<1x1xi8>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.zeros_ui8(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],ui8> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.zeros %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],ui8>
  return %1 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_f32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<1.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] : tensor<f32> into tensor<1x1xf32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.ones_f32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.ones %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_si32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],si32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i32> into tensor<1x1xi32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.ones_si32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],si32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.ones %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],si32>
  return %1 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_ui8(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],ui8> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i8>} : tensor<i8>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i8> into tensor<1x1xi8>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.ones_ui8(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],ui8> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.ones %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],ui8>
  return %1 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_like_f32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<0.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] : tensor<f32> into tensor<1x1xf32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.zeros_like_f32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
%int0 = torch.constant.int 0
%int3 = torch.constant.int 3
%false = torch.constant.bool false
%none = torch.constant.none
%cuda3A0 = torch.constant.device "cuda:0"
%0 = torch.aten.zeros_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_like_si32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i32> into tensor<1x1xi32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.zeros_like_si32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
%int0 = torch.constant.int 0
%int3 = torch.constant.int 3
%false = torch.constant.bool false
%none = torch.constant.none
%cuda3A0 = torch.constant.device "cuda:0"
%0 = torch.aten.zeros_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],si32>
return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_like_ui8(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i8>} : tensor<i8>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i8> into tensor<1x1xi8>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.zeros_like_ui8(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
%int0 = torch.constant.int 0
%int3 = torch.constant.int 3
%false = torch.constant.bool false
%none = torch.constant.none
%cuda3A0 = torch.constant.device "cuda:0"
%0 = torch.aten.zeros_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],ui8>
return %0 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_like_f32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<1.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] : tensor<f32> into tensor<1x1xf32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.ones_like_f32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
%int0 = torch.constant.int 0
%int3 = torch.constant.int 3
%false = torch.constant.bool false
%none = torch.constant.none
%cuda3A0 = torch.constant.device "cuda:0"
%0 = torch.aten.ones_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_like_si32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i32> into tensor<1x1xi32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.ones_like_si32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
%int0 = torch.constant.int 0
%int3 = torch.constant.int 3
%false = torch.constant.bool false
%none = torch.constant.none
%cuda3A0 = torch.constant.device "cuda:0"
%0 = torch.aten.ones_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],si32>
return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_like_ui8(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i8>} : tensor<i8>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] : tensor<i8> into tensor<1x1xi8>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.ones_like_ui8(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
%int0 = torch.constant.int 0
%int3 = torch.constant.int 3
%false = torch.constant.bool false
%none = torch.constant.none
%cuda3A0 = torch.constant.device "cuda:0"
%0 = torch.aten.ones_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],ui8>
return %0 : !torch.vtensor<[?,?],ui8>
}
