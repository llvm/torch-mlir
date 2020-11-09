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

// CHECK-LABEL:   func @tcp_splatted(
// CHECK-SAME:                       %[[SPLAT_VAL:.*]]: f32,
// CHECK-SAME:                       %[[SHAPE:.*]]: tensor<?xindex>) -> tensor<?x?xf32> {
// CHECK:           %[[RESULT:.*]] = refback.alloc_memref %[[SHAPE]] : memref<?x?xf32>
// CHECK:           linalg.fill(%[[RESULT]], %[[SPLAT_VAL]]) : memref<?x?xf32>, f32
// CHECK:           %[[RESULT_TENSOR:.*]] = tensor_load %[[RESULT]] : memref<?x?xf32>
// CHECK:           return %[[RESULT_TENSOR]] : tensor<?x?xf32>
func @tcp_splatted(%arg0: f32, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %0 = tcp.splatted %arg0, %arg1 : (f32, tensor<?xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
