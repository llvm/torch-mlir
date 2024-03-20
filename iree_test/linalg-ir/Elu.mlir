#map = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant dense<0> : tensor<i64>
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%cst : tensor<i64>) outs(%0 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %8 = arith.sitofp %in : i64 to f32
      linalg.yield %8 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%1, %arg0 : tensor<f32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.cmpf ogt, %in, %in_1 : f32
      %9 = arith.select %8, %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%1, %arg0 : tensor<f32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.cmpf olt, %in, %in_1 : f32
      %9 = arith.select %8, %in, %in_1 : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.exp %in : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.subf %in, %cst_0 : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%3, %6 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.addf %in, %in_1 : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    return %7 : tensor<128x128xf32>
  }
}
