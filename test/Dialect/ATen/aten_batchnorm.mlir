// RUN: npcomp-opt %s -aten-layer-name -aten-op-report |& FileCheck %s
//   CHECK-LABEL:       "L0-batch_norm-0": {
//   CHECK-NEXT:          "activation_in": 103320,
//   CHECK-NEXT:          "activation_out": 103320,
//   CHECK-NEXT:          "ops:*": 310206,
//   CHECK-NEXT:          "ops:+": 413280,
//   CHECK-NEXT:          "ops:-": 123,
//   CHECK-NEXT:          "ops:/": 123,
//   CHECK-NEXT:          "ops:sqrt": 123,
//   CHECK-NEXT:          "parameters_in": 246,
//   CHECK-NEXT:          "reads": 103566,
//   CHECK-NEXT:          "writes": 103320

module {
  func @graph(%arg0: tensor<42x123x4x5xf32>, %arg1: tensor<123xf32>, %arg2: tensor<123xf32>, %arg3: tensor<123xf32>, %arg4: tensor<123xf32>, %arg5: tensor<?xi64>) -> tensor<42x123x4x5xf32> {
    %0 = "aten.constant"() {type = "bool", value = 0 : i1} : () -> i1
    %1 = "aten.constant"() {type = "f32", value = 1.000000e-01 : f32} : () -> f32
    %2 = "aten.constant"() {type = "f32", value = 9.99999974E-6 : f32} : () -> f32
    %3 = "aten.constant"() {type = "bool", value = 1 : i1} : () -> i1
    %4:3 = "aten.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %1, %2, %3) : (tensor<42x123x4x5xf32>, tensor<123xf32>, tensor
<123xf32>, tensor<123xf32>, tensor<123xf32>, i1, f32, f32, i1) -> (tensor<42x123x4x5xf32>, tensor<123xf32>, tensor<123xf32>)
    return %4#0 : tensor<42x123x4x5xf32>
  }
}
