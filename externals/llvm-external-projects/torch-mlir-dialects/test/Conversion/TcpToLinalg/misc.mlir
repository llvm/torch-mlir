// RUN: torch-mlir-dialects-opt <%s -convert-tcp-to-linalg -split-input-file | FileCheck %s

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @broadcast_2D(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<1x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: index) -> tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<1x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[ARG1]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]] :  tensor<1x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           linalg.yield %[[BBARG0]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @broadcast_2D(%arg0 : tensor<1x?xf32>, %arg1 : index) -> tensor<?x?xf32> {
  %0 = "tcp.broadcast"(%arg0, %arg1) {axes = [0]} : (tensor<1x?xf32>, index) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @broadcast_4D(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x1x?x1xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: index,
// CHECK-SAME:          %[[ARG2:.*]]: index) -> tensor<?x?x?x?xf32>
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x1x?x1xf32>
// CHECK:         %[[CONST2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[CONST2]] : tensor<?x1x?x1xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[ARG1]], %[[DIM2]], %[[ARG2]]) : tensor<?x?x?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]] : tensor<?x1x?x1xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?x?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           linalg.yield %[[BBARG0]] : f32
// CHECK:         } -> tensor<?x?x?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?x?x?xf32>
// CHECK:       }
func.func @broadcast_4D(%arg0 : tensor<?x1x?x1xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?xf32> {
  %0 = "tcp.broadcast"(%arg0, %arg1, %arg2) {axes = [1, 3]} : (tensor<?x1x?x1xf32>, index, index) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
