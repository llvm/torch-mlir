// RUN: npcomp-opt <%s -convert-tcf-to-linalg | FileCheck %s --dump-input=fail

// CHECK-LABEL:   func @tcf_matmul(
// CHECK-SAME:                     %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                     %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[LHSK:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[RHSK:.*]] = tensor.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[KEQUAL:.*]] = cmpi eq, %[[LHSK]], %[[RHSK]] : index
// CHECK:           %[[WINESS:.*]] = shape.cstr_require %[[KEQUAL]], "mismatching contracting dimension for matmul"
// CHECK:           %[[RET:.*]] = shape.assuming %[[WINESS]] -> (tensor<?x?xf32>) {
// CHECK:             %[[LHSROWS:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:             %[[RHSCOLS:.*]] = tensor.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:             %[[SHAPE:.*]] = tensor.from_elements %[[LHSROWS]], %[[RHSCOLS]] : tensor<2xindex>
// CHECK:             %[[INIT_TENSOR:.*]] = tcp.splatted %[[C0F32]], %[[SHAPE]] : (f32, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[INIT_TENSOR]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[MATMUL]] : tensor<?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?xf32>
func @tcf_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcf.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @tcf_conv_2d_nchw(
// CHECK-SAME:                     %[[IN:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:                     %[[FILTER:[a-zA-Z0-9]+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[C0F32:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C2:.*]] = constant 2 : index
// CHECK:           %[[C3:.*]] = constant 3 : index
// CHECK:           %[[CHANNELS:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:           %[[HEIGHT:.*]] = tensor.dim %[[IN]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:           %[[WIDTH:.*]] = tensor.dim %[[IN]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:           %[[FILTERCHANNELS:.*]] = tensor.dim %[[FILTER]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:           %[[FILTERHEIGHT:.*]] = tensor.dim %[[FILTER]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:           %[[FILTERWIDTH:.*]] = tensor.dim %[[FILTER]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:           %[[CMPCHANNELS:.*]] = cmpi eq, %[[CHANNELS]], %[[FILTERCHANNELS]] : index
// CHECK:           %[[CMPHEIGHT:.*]] = cmpi uge, %[[HEIGHT]], %[[FILTERHEIGHT]] : index
// CHECK:           %[[CMPWIDTH:.*]] = cmpi uge, %[[WIDTH]], %[[FILTERWIDTH]] : index
// CHECK:           %[[CSTRCHANNELS:.*]] = shape.cstr_require %[[CMPCHANNELS]], "input and filter in-channels must be equal"
// CHECK:           %[[CSTRHEIGHT:.*]] = shape.cstr_require %[[CMPHEIGHT]], "input height must be greater than or equal to filter KH-dimension"
// CHECK:           %[[CSTRWIDTH:.*]] = shape.cstr_require %[[CMPWIDTH]], "input width must be greater than or equal to filter KW-dimension"
// CHECK:           %[[WITNESS:.*]] = shape.assuming_all %[[CSTRCHANNELS]], %[[CSTRHEIGHT]], %[[CSTRWIDTH]]
// CHECK:           %[[RET:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?x?x?x?xf32>) {
// CHECK:             %[[BATCH:.*]] = tensor.dim %[[IN]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:             %[[HEIGHT:.*]] = tensor.dim %[[IN]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:             %[[WIDTH:.*]] = tensor.dim %[[IN]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:             %[[OUTCHANNELS:.*]] = tensor.dim %[[FILTER]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:             %[[FILTERHEIGHT:.*]] = tensor.dim %[[FILTER]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:             %[[FILTERWIDTH:.*]] = tensor.dim %[[FILTER]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:             %[[FILTERHEIGHTM1:.*]] = subi %[[FILTERHEIGHT]], %[[C1]] : index
// CHECK:             %[[HEIGHTV0:.*]] = subi %[[HEIGHT]], %[[FILTERHEIGHTM1]] : index
// CHECK:             %[[HEIGHTV0M1:.*]] = subi %[[HEIGHTV0]], %[[C1]] : index
// CHECK:             %[[OUTHEIGHT:.*]] = addi %[[HEIGHTV0M1]], %[[C1]] : index
// CHECK:             %[[FILTERWIDTHM1:.*]] = subi %[[FILTERWIDTH]], %[[C1]] : index
// CHECK:             %[[WIDTHV0:.*]] = subi %[[WIDTH]], %[[FILTERWIDTHM1]] : index
// CHECK:             %[[WIDTHV0M1:.*]] = subi %[[WIDTHV0]], %[[C1]] : index
// CHECK:             %[[OUTWIDTH:.*]] = addi %[[WIDTHV0M1]], %[[C1]] : index
// CHECK:             %[[SHAPE:.*]] = tensor.from_elements %[[BATCH]], %[[OUTCHANNELS]], %[[OUTHEIGHT]], %[[OUTWIDTH]] : tensor<4xindex>
// CHECK:             %[[INIT_TENSOR:.*]] = tcp.splatted %[[C0F32]], %[[SHAPE]] : (f32, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:             %[[CONVNCHW:.*]] = linalg.conv_2d_nchw ins(%[[IN]], %[[FILTER]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%[[INIT_TENSOR]] : tensor<?x?x?x?xf32>)  -> tensor<?x?x?x?xf32>
// CHECK:             shape.assuming_yield %[[CONVNCHW]] : tensor<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return %[[RET:.*]] : tensor<?x?x?x?xf32>
func @tcf_conv_2d_nchw(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcf.conv_2d_nchw %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
