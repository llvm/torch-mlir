// RUN: npcomp-opt -lower-shape-constraints <%s | FileCheck %s

func @cstr_broadcastable(%arg0: tensor<?xindex>, %arg1: tensor<?xindex>) -> !shape.witness {
  %witness = shape.cstr_broadcastable %arg0, %arg1 : tensor<?xindex>, tensor<?xindex>
  return %witness : !shape.witness
}
// There's not very much useful to check here other than pasting the output.
// CHECK-LABEL:   func @cstr_broadcastable(
// CHECK-SAME:                             %[[VAL_0:.*]]: tensor<?xindex>,
// CHECK-SAME:                             %[[VAL_1:.*]]: tensor<?xindex>) -> !shape.witness {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = shape.const_witness true
// CHECK:           %[[VAL_5:.*]] = dim %[[VAL_0]], %[[VAL_2]] : tensor<?xindex>
// CHECK:           %[[VAL_6:.*]] = dim %[[VAL_1]], %[[VAL_2]] : tensor<?xindex>
// CHECK:           %[[VAL_7:.*]] = cmpi "ule", %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]]:4 = scf.if %[[VAL_7]] -> (index, tensor<?xindex>, index, tensor<?xindex>) {
// CHECK:             scf.yield %[[VAL_5]], %[[VAL_0]], %[[VAL_6]], %[[VAL_1]] : index, tensor<?xindex>, index, tensor<?xindex>
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_6]], %[[VAL_1]], %[[VAL_5]], %[[VAL_0]] : index, tensor<?xindex>, index, tensor<?xindex>
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = subi %[[VAL_10:.*]]#2, %[[VAL_10]]#0 : index
// CHECK:           scf.for %[[VAL_11:.*]] = %[[VAL_9]] to %[[VAL_10]]#2 step %[[VAL_3]] {
// CHECK:             %[[VAL_12:.*]] = extract_element %[[VAL_10]]#3{{\[}}%[[VAL_11]]] : tensor<?xindex>
// CHECK:             %[[VAL_13:.*]] = subi %[[VAL_11]], %[[VAL_9]] : index
// CHECK:             %[[VAL_14:.*]] = extract_element %[[VAL_10]]#1{{\[}}%[[VAL_13]]] : tensor<?xindex>
// CHECK:             %[[VAL_15:.*]] = cmpi "eq", %[[VAL_12]], %[[VAL_3]] : index
// CHECK:             %[[VAL_16:.*]] = cmpi "eq", %[[VAL_14]], %[[VAL_3]] : index
// CHECK:             %[[VAL_17:.*]] = cmpi "eq", %[[VAL_12]], %[[VAL_14]] : index
// CHECK:             %[[VAL_18:.*]] = or %[[VAL_15]], %[[VAL_16]] : i1
// CHECK:             %[[VAL_19:.*]] = or %[[VAL_17]], %[[VAL_18]] : i1
// CHECK:             assert %[[VAL_19]], "invalid broadcast"
// CHECK:           }
// CHECK:           return %[[VAL_4]] : !shape.witness
// CHECK:         }

// Check that `shape.assuming` is eliminated after we create the error handling code.
// CHECK-LABEL: func @assuming
func @assuming(%arg0: tensor<?xindex>, %arg1: tensor<?xindex>) -> tensor<2xf32> {
  %witness = shape.cstr_broadcastable %arg0, %arg1 : tensor<?xindex>, tensor<?xindex>
  // CHECK-NOT: shape.assuming
  // CHECK: %[[CST:.*]] = constant dense<0.000000e+00> : tensor<2xf32>
  %0 = shape.assuming %witness -> tensor<2xf32> {
    %c = constant dense<0.0> : tensor<2xf32>
    shape.assuming_yield %c : tensor<2xf32>
  }
  // CHECK: return %[[CST]]
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @cstr_require
func @cstr_require(%arg0: i1) -> !shape.witness {
  // CHECK: %[[RET:.*]] = shape.const_witness true
  // CHECK: assert %arg0, "msg"
  // CHECK: return %[[RET]]
  %witness = shape.cstr_require %arg0, "msg"
  return %witness : !shape.witness
}
