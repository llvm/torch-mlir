// RUN: npcomp-opt -tcp-bufferize <%s | FileCheck %s

// CHECK-LABEL:   func @tcp_broadcast_to(
// CHECK-SAME:                           %[[TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:                           %[[SHAPE:.*]]: tensor<?xindex>) -> tensor<?x?xf32> {
// CHECK:           refback.alloc_memref %[[SHAPE]] : memref<?x?xf32>
// Check for two nested loops, but don't look at more detail for now.
// TODO: This pass should not create loops. Instead it should create a
// buffer version of tcp.broadcast_to
// CHECK:           scf.for
// CHECK:             scf.for
func @tcp_broadcast_to(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %0 = tcp.broadcast_to %arg0, %arg1 : (tensor<?xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @tcp_matmul(
// CHECK-SAME:                     %[[LHS_TENSOR:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                     %[[RHS_TENSOR:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[LHS:.*]] = tensor_to_memref %[[LHS_TENSOR]] : memref<?x?xf32>
// CHECK:           %[[RHS:.*]] = tensor_to_memref %[[RHS_TENSOR]] : memref<?x?xf32>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[LHS_ROWS:.*]] = dim %[[LHS_TENSOR]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[RHS_COLS:.*]] = dim %[[RHS_TENSOR]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[SHAPE:.*]] = tensor_from_elements %[[LHS_ROWS]], %[[RHS_COLS]] : tensor<2xindex>
// CHECK:           %[[RESULT:.*]] = refback.alloc_memref %[[SHAPE]] : memref<?x?xf32>
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           linalg.fill(%[[RESULT]], %[[C0F32]]) : memref<?x?xf32>, f32
// CHECK:           linalg.matmul ins(%[[LHS]], %[[RHS]] : memref<?x?xf32>, memref<?x?xf32>) outs(%[[RESULT]] : memref<?x?xf32>)
// CHECK:           %[[RESULT_TENSOR:.*]] = tensor_load %[[RESULT]] : memref<?x?xf32>
// CHECK:           return %[[RESULT_TENSOR]] : tensor<?x?xf32>
// CHECK:         }
func @tcp_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
