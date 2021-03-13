// RUN: npcomp-opt <%s -convert-aten-to-tcf | FileCheck %s --dump-input=fail

// CHECK-LABEL: @conv2d
func @conv2d(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[CONV2D_RESULT:.*]] = tcf.conv_2d_nchw %arg0, %arg1 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK: tcf.add %[[CONV2D_RESULT]], %arg2 : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %c0_i64 = constant 0 : i64
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %1 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %3 = "aten.conv2d"(%arg0, %arg1, %arg2, %0, %1, %2, %c1_i64) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> tensor<?x?x?x?xf32>
  return %3 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @binary_elementwise_ops
// NOTE: These are all template expanded, so just testing an examplar op and
// special cases.
func @binary_elementwise_ops(%arg0: tensor<4x6x1xf32>, %arg1: tensor<1x1x3xf32>) -> tensor<4x6x3xf32> {
  // CHECK: tcf.mul %arg0, %arg1 : (tensor<4x6x1xf32>, tensor<1x1x3xf32>) -> tensor<4x6x3xf32>
  %0 = "aten.mul"(%arg0, %arg1) : (tensor<4x6x1xf32>, tensor<1x1x3xf32>) -> tensor<4x6x3xf32>
  return %0 : tensor<4x6x3xf32>
}

// CHECK-LABEL: @add_alpha_constant1
func @add_alpha_constant1(%arg0: tensor<4x6x3xf32>, %arg1: tensor<1x1x3xf32>) -> tensor<4x6x3xf32> {
  %c1_i64 = constant 1 : i64
  // CHECK: tcf.add %arg0, %arg1 : (tensor<4x6x3xf32>, tensor<1x1x3xf32>) -> tensor<4x6x3xf32>
  %0 = "aten.add"(%arg0, %arg1, %c1_i64) : (tensor<4x6x3xf32>, tensor<1x1x3xf32>, i64) -> tensor<4x6x3xf32>
  return %0 : tensor<4x6x3xf32>
}

// CHECK-LABEL: @add_alpha_non_constant1
func @add_alpha_non_constant1(%arg0: tensor<4x6x3xf32>, %arg1: tensor<1x1x3xf32>) -> tensor<4x6x3xf32> {
  %c1_i64 = constant 2 : i64
  // CHECK: "aten.add"
  %0 = "aten.add"(%arg0, %arg1, %c1_i64) : (tensor<4x6x3xf32>, tensor<1x1x3xf32>, i64) -> tensor<4x6x3xf32>
  return %0 : tensor<4x6x3xf32>
}
