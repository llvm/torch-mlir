// RUN: npcomp-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @torch.aten.mm$basic(
// CHECK-SAME:                        %[[LHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                        %[[RHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
// CHECK:           %[[LHS:.*]] = torch.to_builtin_tensor %[[LHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RHS:.*]] = torch.to_builtin_tensor %[[RHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[LHS_DIM_0:.*]] = memref.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[LHS_DIM_1:.*]] = memref.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[RHS_DIM_0:.*]] = memref.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[RHS_DIM_1:.*]] = memref.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[EQ:.*]] = cmpi eq, %[[LHS_DIM_1]], %[[RHS_DIM_0]] : index
// CHECK:           assert %[[EQ]], "mismatching contracting dimension for torch.aten.mm"
// CHECK:           %[[INIT_TENSOR:.*]] = linalg.init_tensor [%[[LHS_DIM_0]], %[[RHS_DIM_1]]] : tensor<?x?xf32>
// CHECK:           %[[CF0:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[ZEROFILL:.*]] = linalg.fill(%[[INIT_TENSOR]], %[[CF0]]) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
// CHECK:           %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZEROFILL]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[MATMUL]] : tensor<?x?xf32> to tensor<?x2xf32>
// CHECK:           %[[RESULT_VTENSOR:.*]] = torch.from_builtin_tensor %[[CASTED]] : tensor<?x2xf32> -> !torch.vtensor<[?,2],f32>
// CHECK:           return %[[RESULT_VTENSOR]] : !torch.vtensor<[?,2],f32>
func @torch.aten.mm$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}

// Unary op example.
// CHECK-LABEL:   func @torch.aten.tanh(
// CHECK-SAME:                          %[[ARG_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG:.*]] = torch.to_builtin_tensor %[[ARG_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[ARG]] : tensor<?x?xf32>) outs(%[[ARG]] : tensor<?x?xf32>) {
// CHECK:           ^bb0(%[[BBARG:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[YIELDED:.*]] = math.tanh %[[BBARG]] : f32
// CHECK:             linalg.yield %[[YIELDED]] : f32
// CHECK:           } -> tensor<?x?xf32>
// CHECK:           %[[RESULT_VTENSOR:.*]] = torch.from_builtin_tensor %[[RESULT]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT_VTENSOR:.*]] : !torch.vtensor<[?,?],f32>
func @torch.aten.tanh(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// If the operands are missing dtype, we cannot lower it.
func @torch.aten.mm$no_convert$missing_dtype(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// Correctly handle the case that operands are statically the wrong rank
// (rank 1 vs rank 2 expected for matmul.)
func @torch.aten.mm$no_convert$wrong_rank(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// If the result is missing dtype, we cannot lower it.
func @torch.aten.mm$no_convert$result_missing_dtype(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor
  return %0 : !torch.vtensor
}
