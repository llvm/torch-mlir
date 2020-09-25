// RUN: npcomp-opt <%s -convert-tcf-to-tcp | FileCheck %s --dump-input=fail

// CHECK-LABEL:   func @unary_ops(
// CHECK-SAME:                    %[[ARG:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[RET:.*]] = tcp.exp %[[ARG]] : tensor<?xf32>
// CHECK:           return %[[RET]] : tensor<?xf32>
// CHECK:         }
func @unary_ops(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcf.exp %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   func @tcf_add(
// CHECK-SAME:            %[[LHS:.*]]: tensor<?xf32>,
// CHECK-SAME:            %[[RHS:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[LHSSHAPE:.*]] = shape.shape_of %[[LHS]]
// CHECK:           %[[RHSSHAPE:.*]] = shape.shape_of %[[RHS]]
// CHECK:           %[[WITNESS:.*]] = shape.cstr_broadcastable %[[LHSSHAPE]], %[[RHSSHAPE]]
// CHECK:           %[[RET:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?xf32>) {
// CHECK:             %[[RESULTSHAPE:.*]] = shape.broadcast %[[LHSSHAPE]], %[[RHSSHAPE]]
// CHECK:             %[[LHSBCAST:.*]] = tcp.broadcast_to %[[LHS]], %[[RESULTSHAPE]]
// CHECK:             %[[RHSBCAST:.*]] = tcp.broadcast_to %[[RHS]], %[[RESULTSHAPE]]
// CHECK:             %[[ADD:.*]] = tcp.add %[[LHSBCAST]], %[[RHSBCAST]]
// CHECK:             shape.assuming_yield %[[ADD]] : tensor<?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?xf32>
// CHECK:         }
func @tcf_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   func @tcf_matmul(
// CHECK-SAME:                     %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                     %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[LHSK:.*]] = dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[RHSK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[KEQUAL:.*]] = cmpi "eq", %[[LHSK]], %[[RHSK]] : index
// CHECK:           %[[WITNESS:.*]] = shape.cstr_require %[[KEQUAL]], "{{.*}}"
// CHECK:           %[[RET:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?x?xf32>) {
// CHECK:             %[[MATMUL:.*]] = tcp.matmul %[[LHS]], %[[RHS]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[MATMUL]] : tensor<?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?xf32>
// CHECK:         }
func @tcf_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcf.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
