// RUN: npcomp-opt %s -aten-layer-name -aten-to-std |& FileCheck %s
// CHECK: @graph
module {
  func @graph(%arg0: tensor<64x36864xf32>) -> tensor<64x64x24x24xf32> {
    %0 = "aten.constant"() {type = "List[i32]", value = dense<[64, 64, 24, 24]> : vector<4xi32>} : () -> !aten.list<i32>
    %1 = "aten.constant"() {type = "List[i32]", value = dense<[1, 576, 24, 1]> : vector<4xi32>} : () -> !aten.list<i32>
    %2 = "aten.as_strided"(%arg0, %0, %1) {layer_name = "L0-as_strided-0"} : (tensor<64x36864xf32>, !aten.list<i32>, !aten.list<i32>) -> tensor<64x64x24x24xf32>
    return %2 : tensor<64x64x24x24xf32>
  }
}
