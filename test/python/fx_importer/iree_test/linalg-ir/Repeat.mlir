#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64) -> tensor<32768xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<128x128xf32> into tensor<16384xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] : tensor<16384xf32> into tensor<16384x1xf32>
    %0 = tensor.empty() : tensor<16384x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<16384x1xf32>) outs(%0 : tensor<16384x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16384x2xf32>
    %collapsed_0 = tensor.collapse_shape %1 [[0, 1]] : tensor<16384x2xf32> into tensor<32768xf32>
    return %collapsed_0 : tensor<32768xf32>
  }
}
