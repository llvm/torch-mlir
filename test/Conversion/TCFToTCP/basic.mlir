// RUN: npcomp-opt <%s -convert-tcf-to-tcp | FileCheck %s --dump-input=fail

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
