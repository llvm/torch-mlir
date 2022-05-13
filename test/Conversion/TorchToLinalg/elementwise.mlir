// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -mlir-print-local-scope -verify-diagnostics | FileCheck %s


// CHECK-LABEL:   func.func @elementwise$unary(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[INIT_TENSOR:.*]] = linalg.init_tensor [] : tensor<f32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%[[BUILTIN_TENSOR]] : tensor<f32>) outs(%[[INIT_TENSOR]] : tensor<f32>) {
// CHECK:           ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[TANH:.*]] = math.tanh %[[BBARG0]] : f32
// CHECK:             linalg.yield %[[TANH]] : f32
// CHECK:           } -> tensor<f32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[GENERIC:.*]] : tensor<f32> to tensor<f32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[CASTED]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[],f32>
// CHECK:         }
func.func @elementwise$unary(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @elementwise$binary(
// CHECK-SAME:                             %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                             %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[BUILTIN_ARG0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[BUILTIN_ARG1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[ARG0_DIM0:.*]] = tensor.dim %[[BUILTIN_ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[ARG0_DIM1:.*]] = tensor.dim %[[BUILTIN_ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[ARG1_DIM0:.*]] = tensor.dim %[[BUILTIN_ARG1]], %[[C0_2]] : tensor<?xf32>
// CHECK:           %[[LEGAL_SIZES:.*]] = arith.cmpi eq, %[[ARG0_DIM1]], %[[ARG1_DIM0]] : index
// CHECK:           assert %[[LEGAL_SIZES]], "mismatched size for broadcast"
// CHECK:           %[[INIT_TENSOR:.*]] = linalg.init_tensor [%[[ARG0_DIM0]], %[[ARG0_DIM1]]] : tensor<?x?xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%[[BUILTIN_ARG0]], %[[BUILTIN_ARG1]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[INIT_TENSOR]] : tensor<?x?xf32>) {
// CHECK:           ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:             linalg.yield %[[MUL]] : f32
// CHECK:           } -> tensor<?x?xf32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[GENERIC:.*]] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[CASTED]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func.func @elementwise$binary(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// CHECK-LABEL:   func.func @elementwise$ternary(
// CHECK:       linalg.generic {indexing_maps = [
// CHECK-SAME:    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:    affine_map<(d0, d1, d2) -> (d1, d2)>,
// CHECK-SAME:    affine_map<(d0, d1, d2) -> (d2)>,
// CHECK-SAME:    affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
func.func @elementwise$ternary(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.lerp.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL:   func.func @elementwise$with_scalar_capture(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[C1:.*]] = torch.constant.int 1
// CHECK:           %[[BUILTIN_C1:.*]] = torch_c.to_i64 %[[C1]]
// CHECK:           linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>]
// CHECK:           ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[ALPHA:.*]] = arith.sitofp %[[BUILTIN_C1]] : i64 to f32
// CHECK:             %[[SCALED:.*]] = arith.mulf %[[RHS]], %[[ALPHA]] : f32
// CHECK:             %[[RES:.*]] = arith.addf %[[LHS]], %[[SCALED]] : f32
// CHECK:             linalg.yield %[[RES]] : f32
// CHECK:           } -> tensor<?xf32>
func.func @elementwise$with_scalar_capture(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// CHECK-LABEL:   func.func @elementwise$static_1(
// CHECK:           linalg.generic {indexing_maps = [
// CHECK-SAME:        affine_map<(d0) -> (d0)>,
// CHECK-SAME:        affine_map<(d0) -> (0)>,
// CHECK-SAME:        affine_map<(d0) -> (d0)>]
func.func @elementwise$static_1(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[?],f32> {
  %1 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[?],f32>, !torch.vtensor<[1],f32> -> !torch.vtensor<[?],f32>
  return %1 : !torch.vtensor<[?],f32>
}
