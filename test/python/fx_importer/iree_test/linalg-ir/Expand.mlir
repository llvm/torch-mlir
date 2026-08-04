#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64, %arg2: i64, %arg3: i64) -> tensor<12x128x128xf32> {
    %0 = tensor.empty() : tensor<12x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<12x128x128xf32>
    return %1 : tensor<12x128x128xf32>
  }
}
