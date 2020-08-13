// RUN: npcomp-opt %s -aten-to-std |& FileCheck %s --check-prefix=CHECK-CONVERSION
// CHECK-CONVERSION-LABEL: @graph

module {
  func @graph(%arg0: tensor<10xf32>, %arg1: tensor<128xf32>, %arg2: tensor<4x1x28x28xf32>, %arg3: tensor<32x1x3x3xf32>, %arg4: tensor<32xf32>, %arg5: tensor<64x32x3x3xf32>, %arg6: tensor<64xf32>, %arg7: tensor<128x9216xf32>, %arg8: tensor<10x128xf32>) -> tensor<4x10xf32> {
    %0 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %1 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %2 = "aten.constant"() {type = "bool", value = false} : () -> i1
    %3 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %4 = "aten.constant"() {type = "bool", value = true} : () -> i1
    %5 = "aten._convolution"(%arg2, %arg3, %arg4, %0, %1, %0) {layer_name = "L0-_convolution-0"} : (tensor<4x1x28x28xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>) -> tensor<4x32x26x26xf32>
    %6 = "aten.relu"(%5) {layer_name = "L1-relu-0"} : (tensor<4x32x26x26xf32>) -> tensor<4x32x26x26xf32>
    %7 = "aten._convolution"(%6, %arg5, %arg6, %0, %1, %0) {layer_name = "L2-_convolution-1"} : (tensor<4x32x26x26xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>) -> tensor<4x64x24x24xf32>
    %8 = "aten.constant"() {type = "List[i32]", value = dense<2> : vector<2xi32>} : () -> !aten.list<i32>
    %9:2 = "aten.max_pool2d_with_indices"(%7, %8, %8, %1, %0, %2) {layer_name = "L3-max_pool2d_with_indices-0"} : (tensor<4x64x24x24xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, i1) -> (tensor<4x64x12x12xf32>, tensor<4x64x12x12xi64>)
    %10 = "aten.constant"() {type = "List[i32]", value = dense<[4, 9216]> : vector<2xi32>} : () -> !aten.list<i32>
    %11 = "aten.view"(%9#0, %10) {layer_name = "L4-view-0"} : (tensor<4x64x12x12xf32>, !aten.list<i32>) -> tensor<4x9216xf32>
    %12 = "aten.t"(%arg7) {layer_name = "L5-t-0"} : (tensor<128x9216xf32>) -> tensor<9216x128xf32>
    %13 = "aten.addmm"(%arg1, %11, %12, %3, %3) {layer_name = "L6-addmm-0"} : (tensor<128xf32>, tensor<4x9216xf32>, tensor<9216x128xf32>, i32, i32) -> tensor<4x128xf32>
    %14 = "aten.relu"(%13) {layer_name = "L7-relu-1"} : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %15 = "aten.t"(%arg8) {layer_name = "L8-t-1"} : (tensor<10x128xf32>) -> tensor<128x10xf32>
    %16 = "aten.addmm"(%arg0, %14, %15, %3, %3) {layer_name = "L9-addmm-1"} : (tensor<10xf32>, tensor<4x128xf32>, tensor<128x10xf32>, i32, i32) -> tensor<4x10xf32>
    %17 = "aten._log_softmax"(%16, %3, %2) {layer_name = "L10-_log_softmax-0"} : (tensor<4x10xf32>, i32, i1) -> tensor<4x10xf32>
    return %17 : tensor<4x10xf32>
  }
}
