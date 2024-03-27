#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<128x1xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-05 : f64
    %0 = tensor.empty() : tensor<128x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<128x1xf32>) outs(%0 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.truncf %cst_1 : f64 to f32
      %9 = arith.addf %in, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<128x1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x1xf32>) outs(%0 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.sqrt %in : f32
      linalg.yield %8 : f32
    } -> tensor<128x1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x1xf32>) outs(%0 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf one, %in, %cst : f32
      cf.assert %8, "unimplemented: tensor with zero element"
      %9 = arith.divf %cst_0, %in : f32
      linalg.yield %9 : f32
    } -> tensor<128x1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<128x1xf32>) outs(%0 : tensor<128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x1xf32>
    %5 = tensor.empty() : tensor<128x128xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %8 = arith.subf %in, %in_2 : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%6, %4 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %8 = arith.mulf %in, %in_2 : f32
      linalg.yield %8 : f32
    } -> tensor<128x128xf32>
    return %7 : tensor<128x128xf32>
  }
}
