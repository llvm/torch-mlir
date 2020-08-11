// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:  "L0-convolution_backward_overrideable-0": {
//   CHECK-NEXT:    "activation_in": 5568,
//   CHECK-NEXT:    "grad": 5380,
//   CHECK-NEXT:    "ops:+": 768,
//   CHECK-NEXT:    "ops:MAC": 345600,
//   CHECK-NEXT:    "parameters_in": 576,
//   CHECK-NEXT:    "reads": 6144,
//   CHECK-NEXT:    "writes": 5380
//   CHECK-NEXT:  }

// RUN: npcomp-opt %s -aten-to-std |& FileCheck %s --check-prefix=CHECK-CONVERSION
// CHECK-CONVERSION-LABEL: @graph
module {
  func @graph(%arg0: tensor<3x4x8x8xf32>, %arg1: tensor<3x16x10x10xf32>, %arg2: tensor<4x16x3x3xf32>) -> tensor<4x16x3x3xf32> {
    %0 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %1 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %2 = "aten.constant"() {type = "bool", value = false} : () -> i1
    %3 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %10:3 = "aten.convolution_backward_overrideable"(%arg0, %arg1, %arg2, %0, %1, %0, %2, %1, %3) {layer_name = "L5-convolution_backward_overrideable-0"} : (tensor<3x4x8x8xf32>, tensor<3x16x10x10xf32>, tensor<4x16x3x3xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, i1, !aten.list<i32>, i32) -> (tensor<3x16x10x10xf32>, tensor<4x16x3x3xf32>, tensor<4xf32>)
    return %10#1 : tensor<4x16x3x3xf32>
  }
}
