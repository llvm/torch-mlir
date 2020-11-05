// RUN: npcomp-opt <%s -convert-aten-to-tcf | FileCheck %s --dump-input=fail

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
