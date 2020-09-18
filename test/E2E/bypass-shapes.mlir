// RUN: npcomp-opt -bypass-shapes <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @tcp_broadcast_to
func @tcp_broadcast_to(%arg0: tensor<?xf32>, %arg1: tensor<?xindex>) {
  // CHECK: %0 = tcp.shaped_results %arg1
  %0 = "tcp.broadcast_to"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  return
}

// CHECK-LABEL:   func @tcp_add(
// CHECK-SAME:                  %[[LHS:.*]]: tensor<?xf32>,
// CHECK-SAME:                  %[[RHS:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[LHSSHAPE:.*]] = shape.shape_of %[[LHS]]
// CHECK:           %[[RET:.*]] = tcp.shaped_results %[[LHSSHAPE]]
// CHECK:           return %[[RET:.*]] : tensor<?xf32>
// CHECK:         }
func @tcp_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tcp.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// CHECK-LABEL:   func @tcp_matmul(
// CHECK-SAME:                 %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                 %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[LHSCOLS:.*]] = dim %[[LHS]], %[[C0]]
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[RHSROWS:.*]] = dim %[[RHS]], %[[C1]]
// CHECK:           %[[RESULTSHAPE:.*]] = tensor_from_elements %[[LHSCOLS]], %[[RHSROWS]]
// CHECK:           %[[RET:.*]] = tcp.shaped_results %[[RESULTSHAPE]] {
// CHECK:           return %[[RET:.*]] : tensor<?x?xf32>
// CHECK:         }
func @tcp_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
