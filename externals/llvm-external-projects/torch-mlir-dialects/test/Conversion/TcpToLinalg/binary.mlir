// RUN: torch-mlir-dialects-opt <%s -convert-tcp-to-linalg -split-input-file | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add_f32(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]], %[[ARG1]] :  tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[ADDF:.*]] = arith.addf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK:           linalg.yield %[[ADDF]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @add_f32(%arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add_i32(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xi32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xi32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xi32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]], %[[ARG1]] :  tensor<?x?xi32>, tensor<?x?xi32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xi32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: i32, %[[BBARG1:.*]]: i32, %{{.*}}: i32):
// CHECK:           %[[ADDI:.*]] = arith.addi %[[BBARG0]], %[[BBARG1]] : i32
// CHECK:           linalg.yield %[[ADDI]] : i32
// CHECK:         } -> tensor<?x?xi32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xi32>
// CHECK:       }
func.func @add_i32(%arg0 : tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @sub_f32(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]], %[[ARG1]] :  tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[SUBF:.*]] = arith.subf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK:           linalg.yield %[[SUBF]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @sub_f32(%arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sub %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @mul_f32(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]], %[[ARG1]] :  tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[MULF:.*]] = arith.mulf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK:           linalg.yield %[[MULF]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @mul_f32(%arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.mul %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @div_f32(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]], %[[ARG1]] :  tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[MULF:.*]] = arith.divf %[[BBARG0]], %[[BBARG1]] : f32
// CHECK:           linalg.yield %[[MULF]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @div_f32(%arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.divf %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @atan2_f32(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG0]], %[[ARG1]] :  tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %[[BBARG1:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[MULF:.*]] = math.atan2 %[[BBARG0]], %[[BBARG1]] : f32
// CHECK:           linalg.yield %[[MULF]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @atan2_f32(%arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.atan2 %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
