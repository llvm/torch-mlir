// RUN: npcomp-opt -lower-shaped-results-to-memref <%s -split-input-file | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @tcp_broadcast_to
func @tcp_broadcast_to(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // Check for two nested loops, but don't look at more detail for now.
  // TODO: This pass should not create loops. Instead it should create a
  // buffer version of tcp.broadcast_to
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK-NOT: tcp.shaped_results
  %0 = tcp.shaped_results %arg1 {
    %0 = tcp.broadcast_to %arg0, %arg1 : (tensor<?xf32>, tensor<?xindex>) -> tensor<?x?xf32>
    tcp.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL:   func @tcp_add(
// CHECK-SAME:                  %arg0: tensor<?xf32>,
// CHECK-SAME:                  %arg1: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[LHSSHAPE:.*]] = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
// CHECK:           %[[LHS:.*]] = tcp.tensor_to_memref %arg0 : tensor<?xf32> -> memref<?xf32>
// CHECK:           %[[RHS:.*]] = tcp.tensor_to_memref %arg1 : tensor<?xf32> -> memref<?xf32>
// CHECK:           %[[RESULT:.*]] = tcp.alloc_memref %[[LHSSHAPE]] : memref<?xf32>
// CHECK:           linalg.generic {args_in = 2 : i64, args_out = 1 : i64, {{.*}}} %[[LHS]], %[[RHS]], %[[RESULT]] {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           }: memref<?xf32>, memref<?xf32>, memref<?xf32>
// CHECK:           %[[RET:.*]] = tcp.memref_to_tensor %[[RESULT]] : memref<?xf32> -> tensor<?xf32>
// CHECK:           return %[[RET]] : tensor<?xf32>
// CHECK:         }
func @tcp_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %1 = tcp.shaped_results %0 {
    %2 = tcp.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    tcp.yield %2 : tensor<?xf32>
  } : tensor<?xindex> -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----
// Check just the linalg body. The code is otherwise shared with tcp.add.
// CHECK-LABEL: func @tcp_max
// CHECK:           ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %[[DST:.*]]: f32):
// CHECK:             %[[GREATER:.*]] = cmpf "ogt", %[[LHS]], %[[RHS]] : f32
// CHECK:             %[[MAX:.*]] = select %[[GREATER]], %[[LHS]], %[[RHS]] : f32
// CHECK:             linalg.yield %[[MAX]] : f32
func @tcp_max(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %1 = tcp.shaped_results %0 {
    %2 = tcp.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    tcp.yield %2 : tensor<?xf32>
  } : tensor<?xindex> -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL:   func @tcp_matmul(
// CHECK-SAME:                     %arg0: tensor<?x?xf32>,
// CHECK-SAME:                     %arg1: tensor<?x?xf32>,
// CHECK-SAME:                     %[[SHAPE:.*]]: tensor<?xindex>) -> tensor<?x?xf32> {
// CHECK:           %[[LHS:.*]] = tcp.tensor_to_memref %arg0 : tensor<?x?xf32> -> memref<?x?xf32>
// CHECK:           %[[RHS:.*]] = tcp.tensor_to_memref %arg1 : tensor<?x?xf32> -> memref<?x?xf32>
// CHECK:           %[[RESULT:.*]] = tcp.alloc_memref %[[SHAPE]] : memref<?x?xf32>
// CHECK:           linalg.matmul ins(%[[LHS]], %[[RHS]] : memref<?x?xf32>, memref<?x?xf32>) outs(%[[RESULT]] : memref<?x?xf32>)
// CHECK:           %[[RET:.*]] = tcp.memref_to_tensor %[[RESULT]] : memref<?x?xf32> -> tensor<?x?xf32>
// CHECK:           return %[[RET]] : tensor<?x?xf32>
// CHECK:         }
func @tcp_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %shape: tensor<?xindex>) -> tensor<?x?xf32> {
  %0 = tcp.shaped_results %shape {
    %matmul = tcp.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    tcp.yield %matmul : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
