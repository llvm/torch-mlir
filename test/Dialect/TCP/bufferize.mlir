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
// CHECK:           linalg.fill(%[[SPLAT_VAL]], %[[RESULT]]) : f32, memref<?x?xf32>
// CHECK:           %[[RESULT_TENSOR:.*]] = memref.tensor_load %[[RESULT]] : memref<?x?xf32>
// CHECK:           return %[[RESULT_TENSOR]] : tensor<?x?xf32>
func @tcp_splatted(%arg0: f32, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %0 = tcp.splatted %arg0, %arg1 : (f32, tensor<?xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @tcp_pad(
// CHECK-SAME:                  %[[TENSOR:[a-zA-Z0-9]+]]: tensor<?xf32>,
// CHECK-SAME:                  %[[LOWER_EXPANSION:[a-zA-Z0-9]+]]: tensor<?xindex>,
// CHECK-SAME:                  %[[UPPER_EXPANSION:[a-zA-Z0-9]+]]: tensor<?xindex>,
// CHECK-SAME:                  %[[FILL_VAL:[a-zA-Z0-9]+]]: f32) -> tensor<?xf32> {
// CHECK:           %[[TENSOR_MREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<?xf32>
// CHECK:           %[[LOWER_EXPANSION_MREF:.*]] = memref.buffer_cast %[[LOWER_EXPANSION]] : memref<?xindex>
// CHECK:           %[[UPPER_EXPANSION_MREF:.*]] = memref.buffer_cast %[[UPPER_EXPANSION]] : memref<?xindex>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[LOWER_EXTENT_D1:.*]] = tensor.extract %[[LOWER_EXPANSION]][%[[C0]]] : tensor<?xindex>
// CHECK:           %[[UPPER_EXTENT_D1:.*]] = tensor.extract %[[UPPER_EXPANSION]][%[[C0]]] : tensor<?xindex>
// CHECK:           %[[C0_0:.*]] = constant 0 : index
// CHECK:           %[[D1:.*]] = tensor.dim %[[TENSOR]], %[[C0_0]] : tensor<?xf32>
// CHECK:           %[[D1_EXPANSION:.*]] = addi %[[LOWER_EXTENT_D1]], %[[UPPER_EXTENT_D1]] : index
// CHECK:           %[[D1_OUT:.*]] = addi %[[D1_EXPANSION]], %[[D1]] : index
// CHECK:           %[[D1_OUT_TENSOR:.*]] = tensor.from_elements %[[D1_OUT]] : tensor<1xindex>
// CHECK:           %[[D1_OUT_MREF:.*]] = refback.alloc_memref %[[D1_OUT_TENSOR]] : memref<?xf32>
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[C0_1:.*]] = constant 0 : index
// CHECK:           %[[LOWER_EXTENT_D1_1:.*]] = tensor.extract %[[LOWER_EXPANSION]][%[[C0_1]]] : tensor<?xindex>
// CHECK:           %[[C0_2:.*]] = constant 0 : index
// CHECK:           %[[D1_1:.*]] = tensor.dim %[[TENSOR]], %[[C0_2]] : tensor<?xf32>
// CHECK:           linalg.fill(%[[FILL_VAL]], %[[D1_OUT_MREF]]) : f32, memref<?xf32>
// CHECK:           %[[SUBVIEW:.*]] = memref.subview %[[D1_OUT_MREF]][%[[LOWER_EXTENT_D1_1]]] [%[[D1_1]]] [%[[C1]]] : memref<?xf32> to memref<?xf32, #map>
// CHECK:           linalg.copy(%0, %[[SUBVIEW]]) : memref<?xf32>, memref<?xf32, #map>
// CHECK:           %[[RESULT_TENSOR:.*]] = memref.tensor_load %[[D1_OUT_MREF]] : memref<?xf32>
// CHECK:           return %[[RESULT_TENSOR]] : tensor<?xf32>
func @tcp_pad(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>, %arg2: tensor<?xindex>, %arg3: f32) -> tensor<?xf32> {
  %0 = tcp.pad %arg0, %arg1, %arg2, %arg3 : (tensor<?xf32>, tensor<?xindex>, tensor<?xindex>, f32) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
