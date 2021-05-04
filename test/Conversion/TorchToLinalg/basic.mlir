// RUN: npcomp-opt <%s -convert-torch-to-linalg | FileCheck %s

// CHECK-LABEL:   func @torch.aten.mm$basic(
// CHECK-SAME:                        %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                        %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x2xf32> {
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[CF0:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[LHS_DIM_0:.*]] = memref.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[LHS_DIM_1:.*]] = memref.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[RHS_DIM_0:.*]] = memref.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[RHS_DIM_1:.*]] = memref.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[EQ:.*]] = cmpi eq, %[[LHS_DIM_1]], %[[RHS_DIM_0]] : index
// CHECK:           assert %[[EQ]], "mismatching contracting dimension for torch.aten.mm"
// CHECK:           %[[INIT_TENSOR:.*]] = linalg.init_tensor [%[[LHS_DIM_0]], %[[RHS_DIM_1]]] : tensor<?x?xf32>
// CHECK:           %[[ZEROFILL:.*]] = linalg.fill(%[[INIT_TENSOR]], %[[CF0]]) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
// CHECK:           %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZEROFILL]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[MATMUL]] : tensor<?x?xf32> to tensor<?x2xf32>
// CHECK:           return %[[CASTED]] : tensor<?x2xf32>
func @torch.aten.mm$basic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x2xf32> {
  %0 = torch.aten.mm %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// If the operands are missing dtype, we cannot lower it.
// CHECK-LABEL: func @torch.aten.mm$no_convert$missing_dtype
func @torch.aten.mm$no_convert$missing_dtype(%arg0: tensor<*x!numpy.any_dtype>, %arg1: tensor<*x!numpy.any_dtype>) -> tensor<*x!numpy.any_dtype> {
  // CHECK-NEXT: torch.aten.mm
  %0 = torch.aten.mm %arg0, %arg1 : tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype> -> tensor<*x!numpy.any_dtype>
  return %0 : tensor<*x!numpy.any_dtype>
}

// Correctly handle the case that operands are statically the wrong rank
// (rank 1 vs rank 2 expected for matmul.)
// CHECK-LABEL: func @torch.aten.mm$no_convert$wrong_rank
func @torch.aten.mm$no_convert$wrong_rank(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
  // CHECK-NEXT: torch.aten.mm
  %0 = torch.aten.mm %arg0, %arg1 : tensor<?xf32>, tensor<?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// If the result is missing dtype, we cannot lower it.
// CHECK-LABEL: func @torch.aten.mm$no_convert$result_missing_dtype
func @torch.aten.mm$no_convert$result_missing_dtype(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<*x!numpy.any_dtype> {
  // CHECK-NEXT: torch.aten.mm
  %0 = torch.aten.mm %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<*x!numpy.any_dtype>
  return %0 : tensor<*x!numpy.any_dtype>
}

// Unary op example.
// CHECK-LABEL:   func @torch.aten.tanh(
// CHECK-SAME:                          %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[RESULT:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[ARG]] : tensor<?x?xf32>) outs(%[[ARG]] : tensor<?x?xf32>) {
// CHECK:           ^bb0(%[[BBARG:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[YIELDED:.*]] = math.tanh %[[BBARG]] : f32
// CHECK:             linalg.yield %[[YIELDED]] : f32
// CHECK:           } -> tensor<?x?xf32>
// CHECK:           return %[[RESULT:.*]] : tensor<?x?xf32>
func @torch.aten.tanh(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = torch.aten.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
