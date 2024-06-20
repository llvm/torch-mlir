#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %arg0 : tensor<f32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.cmpf olt, %in, %in_0 : f32
      %10 = arith.select %9, %in, %in_0 : f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.absf %in : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.negf %in : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.exp %in : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = math.log1p %in : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %7 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.subf %in, %in_0 : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    return %8 : tensor<128x128xf32>
  }
}
