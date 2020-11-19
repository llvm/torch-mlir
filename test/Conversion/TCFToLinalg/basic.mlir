// RUN: npcomp-opt <%s -convert-tcf-to-linalg | FileCheck %s --dump-input=fail

// CHECK-LABEL:   func @tcf_matmul(
// CHECK-SAME:                     %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                     %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[LHSK:.*]] = dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[RHSK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[KEQUAL:.*]] = cmpi "eq", %[[LHSK]], %[[RHSK]] : index
// CHECK:           %[[WINESS:.*]] = shape.cstr_require %[[KEQUAL]], "mismatching contracting dimension for matmul"
// CHECK:           %[[RET:.*]] = shape.assuming %[[WINESS]] -> (tensor<?x?xf32>) {
// CHECK:             %[[LHSROWS:.*]] = dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:             %[[RHSCOLS:.*]] = dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:             %[[SHAPE:.*]] = tensor_from_elements %[[LHSROWS]], %[[RHSCOLS]] : tensor<2xindex>
// CHECK:             %[[INIT_TENSOR:.*]] = tcp.splatted %[[C0F32]], %[[SHAPE]] : (f32, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) init(%[[INIT_TENSOR]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[MATMUL]] : tensor<?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?xf32>
func @tcf_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcf.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @tcf_conv_2d_nchw_bias(
// CHECK-SAME:                     %[[IN:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:                     %[[FILTER:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:                     %[[BIAS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[WITNESS:.*]] = shape.const_witness true
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[C2:.*]] = constant 2 : index
// CHECK:           %[[C3:.*]] = constant 3 : index
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[RET:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?x?x?x?xf32>) {
// CHECK:             %[[BATCHK:.*]] = dim %[[IN]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:             %[[HEIGHT:.*]] = dim %[[IN]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:             %[[WIDTH:.*]] = dim %[[IN]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:             %[[FILTERK:.*]] = dim %[[FILTER]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:             %[[SHAPE:.*]] = tensor_from_elements %[[BATCHK]], %[[FILTERK]], %[[HEIGHT]], %[[WIDTH]] : tensor<4xindex>
// CHECK:             %[[INIT_TENSOR:.*]] = tcp.splatted %[[C0F32]], %[[SHAPE]] : (f32, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:             %[[CONVNCHW:.*]] = linalg.conv_2d_nchw ins(%[[IN]], %[[FILTER]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) init(%[[INIT_TENSOR]] : tensor<?x?x?x?xf32>)  -> tensor<?x?x?x?xf32>
// CHECK:             %[[CONVNCHWBIAS:.*]] = addf %[[CONVNCHW]], %[[BIAS]] : tensor<?x?x?x?xf32>
// CHECK:             shape.assuming_yield %[[CONVNCHWBIAS]] : tensor<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?x?x?xf32>
func @tcf_conv_2d_nchw_bias(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcf.conv_2d_nchw_bias %arg0, %arg1, %arg2 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
