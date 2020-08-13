// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:     "L1-addmm-0": {
//   CHECK-NEXT:        "activation_in": 1024,
//   CHECK-NEXT:        "activation_out": 16,
//   CHECK-NEXT:        "ops:+": 16,
//   CHECK-NEXT:        "ops:MAC": 16384,
//   CHECK-NEXT:        "parameters_in": 16400,
//   CHECK-NEXT:        "reads": 17424,
//   CHECK-NEXT:        "writes": 16
//

module {
  func @graph(%arg0: tensor<1x1024xf32>, %arg1: tensor<16x1024xf32>, %arg2: tensor<16xf32>) -> tensor<1x16xf32> {
    %0 = "aten.t"(%arg1) : (tensor<16x1024xf32>) -> tensor<1024x16xf32>
    %1 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %2 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
    %3 = "aten.addmm"(%arg2, %arg0, %0, %1, %2) : (tensor<16xf32>, tensor<1x1024xf32>, tensor<1024x16xf32>, i32, i32) -> tensor<1x16xf32>
    "std.return"(%3) : (tensor<1x16xf32>) -> ()
  }
}
