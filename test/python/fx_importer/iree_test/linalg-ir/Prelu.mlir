#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %expanded = tensor.expand_shape %arg1 [[0, 1]] : tensor<128xf32> into tensor<1x128xf32>
    %0 = tensor.empty() : tensor<128x128xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %5 = arith.cmpf ugt, %in, %cst : f32
      linalg.yield %5 : i1
    } -> tensor<128x128xi1>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded, %arg0 : tensor<1x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %arg0, %3 : tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128xf32>
    return %4 : tensor<128x128xf32>
  }
}
