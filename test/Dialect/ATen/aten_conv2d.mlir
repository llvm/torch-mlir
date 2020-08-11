// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:   "L0-_convolution-0": {
//   CHECK-NEXT:     "activation_in": 32768,
//   CHECK-NEXT:     "activation_out": 65536,
//   CHECK-NEXT:     "ops:+": 65536,
//   CHECK-NEXT:     "ops:MAC": 6422528,
//   CHECK-NEXT:     "parameters_in": 1584,
//   CHECK-NEXT:     "reads": 34352,
//   CHECK-NEXT:     "writes": 65536

module {
  func @graph(%arg0: tensor<1x2x128x128xf32>, %arg1: tensor<16x2x7x7xf32>, %arg2: tensor<16xf32>) -> tensor<1x16x64x64xf32> {
    %0 = "aten.constant"() {type = "List[i32]", value = dense<2> : vector<2xi64>} : () -> !aten.list<i32>
    %1 = "aten.constant"() {type = "List[i32]", value = dense<3> : vector<2xi64>} : () -> !aten.list<i32>
    %2 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi64>} : () -> !aten.list<i32>
    %3 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %4 = "aten.constant"() {type = "List[i32]", value = dense<0> : vector<2xi64>} : () -> !aten.list<i32>
    %5 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %6 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %7 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %8 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %9 = "aten._convolution"(%arg0, %arg1, %arg2, %0, %1, %2) : (tensor<1x2x128x128xf32>, tensor<16x2x7x7xf32>, tensor<16xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>) -> tensor<1x16x64x64xf32>
    "std.return"(%9) : (tensor<1x16x64x64xf32>) -> ()
  }
}
