#map = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xi1>, %arg2: i64) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<2.000000e+00> : tensor<f64>
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%cst : tensor<f64>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %4 = arith.truncf %in : f64 to f32
      linalg.yield %4 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %1, %arg0 : tensor<128x128xi1>, tensor<f32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %4 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<128x128xf32>
    return %3, %3 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
