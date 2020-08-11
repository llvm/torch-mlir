// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:  "L0-relu-0": {
//   CHECK-NEXT:    "activation_in": 6,
//   CHECK-NEXT:    "activation_out": 6,
//   CHECK-NEXT:    "ops:>": 6,
//   CHECK-NEXT:    "reads": 6,
//   CHECK-NEXT:    "writes": 6

module {
  func @graph(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    %0 = "aten.relu"(%arg0) : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    "std.return"(%0) : (tensor<1x2x3xf32>) -> ()
  }
}

