#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64) -> tensor<128x128xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<128xi64>) -> tensor<128xi64>
    %2 = tensor.empty() : tensor<128xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%3, %1 : tensor<128xf32>, tensor<128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %13 = linalg.index 0 : index
      %14 = arith.index_cast %13 : index to i64
      %15 = arith.maximumf %in, %out : f32
      %16 = arith.cmpf ogt, %in, %out : f32
      %17 = arith.select %16, %14, %out_1 : i64
      linalg.yield %15, %17 : f32, i64
    } -> (tensor<128xf32>, tensor<128xi64>)
    %expanded = tensor.expand_shape %4#0 [[0, 1]] : tensor<128xf32> into tensor<1x128xf32>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<128x128xf32>, tensor<1x128xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.subf %in, %in_1 : f32
      linalg.yield %13 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x128xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = math.exp %in : f32
      linalg.yield %13 : f32
    } -> tensor<128x128xf32>
    %8 = tensor.empty() : tensor<1x128xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<1x128xf32>) -> tensor<1x128xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "parallel"]} ins(%7 : tensor<128x128xf32>) outs(%9 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.addf %in, %out : f32
      linalg.yield %13 : f32
    } -> tensor<1x128xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x128xf32>) outs(%8 : tensor<1x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = math.log %in : f32
      linalg.yield %13 : f32
    } -> tensor<1x128xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %11 : tensor<128x128xf32>, tensor<1x128xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.subf %in, %in_1 : f32
      linalg.yield %13 : f32
    } -> tensor<128x128xf32>
    return %12 : tensor<128x128xf32>
  }
}
