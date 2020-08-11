// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:     "L0-native_batch_norm-0": {
//   CHECK-LABEL:     "L1-relu-0": {
//   CHECK-LABEL:     "L2-_convolution-0": {
//   CHECK-LABEL:     "L3-native_batch_norm-1": {
//   CHECK-LABEL:     "L4-relu-1": {
//   CHECK-LABEL:     "L5-_convolution-1": {
//   CHECK-LABEL:     "L6-native_batch_norm-2": {
//   CHECK-LABEL:     "L7-relu-2": {
//   CHECK-LABEL:     "L8-_convolution-2": {
//   CHECK-LABEL:     "L9-add-0": {

module {
  func @graph(%arg0: tensor<1x16x128x128xf32>, %arg1: tensor<1x16x128x128xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>, %arg5: tensor<16xf32>, %arg6: tensor<8x16x1x1xf32>, %arg7: tensor<8xf32>, %arg8: tensor<8xf32>, %arg9: tensor<8xf32>, %arg10: tensor<8xf32>, %arg11: tensor<8xf32>, %arg12: tensor<8x8x3x3xf32>, %arg13: tensor<8xf32>, %arg14: tensor<8xf32>, %arg15: tensor<8xf32>, %arg16: tensor<8xf32>, %arg17: tensor<8xf32>, %arg18: tensor<16x8x1x1xf32>, %arg19: tensor<16xf32>) -> tensor<1x16x128x128xf32> {
    %0 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %1 = "aten.constant"() {type = "f32", value = 1.000000e-01 : f32} : () -> f32
    %2 = "aten.constant"() {type = "f32", value = 9.99999974E-6 : f32} : () -> f32
    %3:3 = "aten.native_batch_norm"(%arg1, %arg2, %arg3, %arg4, %arg5, %0, %1, %2) : (tensor<1x16x128x128xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i1, f32, f32) -> (tensor<1x16x128x128xf32>, tensor<16xf32>, tensor<16xf32>)
    %4 = "aten.relu"(%3#0) : (tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
    %5 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %6 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %7 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %8 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %9 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %10 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %11 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %12 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %13 = "aten._convolution"(%4, %arg6, %arg7, %5, %6, %7) : (tensor<1x16x128x128xf32>, tensor<8x16x1x1xf32>, tensor<8xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>) -> tensor<1x8x128x128xf32>
    %14 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %15 = "aten.constant"() {type = "f32", value = 1.000000e-01 : f32} : () -> f32
    %16 = "aten.constant"() {type = "f32", value = 9.99999974E-6 : f32} : () -> f32
    %17:3 = "aten.native_batch_norm"(%13, %arg8, %arg9, %arg10, %arg11, %14, %15, %16) : (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, i1, f32, f32) -> (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>)
    %18 = "aten.relu"(%17#0) : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %19 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %20 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %21 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %22 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %23 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %24 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %25 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %26 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %27 = "aten._convolution"(%18, %arg12, %arg13, %19, %20, %21) : (tensor<1x8x128x128xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>) -> tensor<1x8x128x128xf32>
    %28 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %29 = "aten.constant"() {type = "f32", value = 1.000000e-01 : f32} : () -> f32
    %30 = "aten.constant"() {type = "f32", value = 9.99999974E-6 : f32} : () -> f32
    %31:3 = "aten.native_batch_norm"(%27, %arg14, %arg15, %arg16, %arg17, %28, %29, %30) : (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, i1, f32, f32) -> (tensor<1x8x128x128xf32>, tensor<8xf32>, tensor<8xf32>)
    %32 = "aten.relu"(%31#0) : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %33 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %34 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %35 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi32>} : () -> !aten.list<i32>
    %36 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %37 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi32>} : () -> !aten.list<i32>
    %38 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %39 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %40 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %41 = "aten._convolution"(%32, %arg18, %arg19, %33, %34, %35) : (tensor<1x8x128x128xf32>, tensor<16x8x1x1xf32>, tensor<16xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>) -> tensor<1x16x128x128xf32>
    %42 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %43 = "aten.add"(%arg0, %41, %42) : (tensor<1x16x128x128xf32>, tensor<1x16x128x128xf32>, i32) -> tensor<1x16x128x128xf32>
    return %43 : tensor<1x16x128x128xf32>
  }
}
