#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: i64, %arg3: i64) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = math.exp %in : f32
      linalg.yield %7 : f32
    } -> tensor<128x128xf32>
    %2 = tensor.empty() : tensor<1x128xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%3 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.addf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<1x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %4 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.mulf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %5 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.subf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<128x128xf32>
    return %6 : tensor<128x128xf32>
  }
}
