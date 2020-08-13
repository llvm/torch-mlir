// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:  "L0-max_pool2d-0": {
//   CHECK-NEXT:    "activation_in": 8192,
//   CHECK-NEXT:    "activation_out": 2048,
//   CHECK-NEXT:    "ops:>": 16384,
//   CHECK-NEXT:    "reads": 8192,
//   CHECK-NEXT:    "writes": 2048

module {
  func @graph(%arg0: tensor<1x32x16x16xf32>) -> tensor<1x32x8x8xf32> {
    %0 = "aten.constant"() {type = "List[i32]", value = dense<3> : vector<2xi64>} : () -> !aten.list<i32>
    %1 = "aten.constant"() {type = "List[i32]", value = dense<2> : vector<2xi64>} : () -> !aten.list<i32>
    %2 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi64>} : () -> !aten.list<i32>
    %3 = "aten.constant"() {type = "List[i32]", value = dense<1> : vector<2xi64>} : () -> !aten.list<i32>
    %4 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %5 = "aten.max_pool2d"(%arg0, %0, %1, %2, %3, %4) : (tensor<1x32x16x16xf32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, !aten.list<i32>, i1) -> tensor<1x32x8x8xf32>
    "std.return"(%5) : (tensor<1x32x8x8xf32>) -> ()
  }
}
