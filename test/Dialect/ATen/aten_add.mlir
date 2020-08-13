// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:     "L0-add-0": {
//   CHECK-NEXT:        "activation_in": 12,
//   CHECK-NEXT:        "activation_out": 6,
//   CHECK-NEXT:        "ops:+": 6,
//   CHECK-NEXT:        "reads": 12,
//   CHECK-NEXT:        "writes": 6

// RUN: npcomp-opt %s -aten-to-std |& FileCheck %s --check-prefix=CHECK-CONVERSION
// CHECK-CONVERSION-LABEL: @graph
func @graph(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %1 = "aten.constant"() {type = "i32", value = 1 : i32} : () -> i32
  %2 = "aten.add"(%arg0, %arg1, %1) : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, i32) -> tensor<1x2x3xf32>
  "std.return"(%2) : (tensor<1x2x3xf32>) -> ()
}
