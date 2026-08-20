#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<() -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<f32> {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.638300e+04 : f64
    %cst_1 = arith.constant 1.638400e+04 : f64
    %0 = tensor.empty() : tensor<128x128xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf64>) {
    ^bb0(%in: f32, %out: f64):
      %14 = arith.extf %in : f32 to f64
      linalg.yield %14 : f64
    } -> tensor<128x128xf64>
    %2 = tensor.empty() : tensor<1x1xf64>
    %3 = linalg.fill ins(%cst : f64) outs(%2 : tensor<1x1xf64>) -> tensor<1x1xf64>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%1 : tensor<128x128xf64>) outs(%3 : tensor<1x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %14 = arith.addf %in, %out : f64
      linalg.yield %14 : f64
    } -> tensor<1x1xf64>
    %5 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x1xf64>) outs(%2 : tensor<1x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %14 = arith.divf %in, %cst_1 : f64
      linalg.yield %14 : f64
    } -> tensor<1x1xf64>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<128x128xf64>, tensor<1x1xf64>) outs(%0 : tensor<128x128xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %14 = arith.subf %in, %in_2 : f64
      linalg.yield %14 : f64
    } -> tensor<128x128xf64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %6 : tensor<128x128xf64>, tensor<128x128xf64>) outs(%0 : tensor<128x128xf64>) {
    ^bb0(%in: f64, %in_2: f64, %out: f64):
      %14 = arith.mulf %in, %in_2 : f64
      linalg.yield %14 : f64
    } -> tensor<128x128xf64>
    %8 = tensor.empty() : tensor<f64>
    %9 = linalg.fill ins(%cst : f64) outs(%8 : tensor<f64>) -> tensor<f64>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "reduction"]} ins(%7 : tensor<128x128xf64>) outs(%9 : tensor<f64>) {
    ^bb0(%in: f64, %out: f64):
      %14 = arith.addf %in, %out : f64
      linalg.yield %14 : f64
    } -> tensor<f64>
    %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%10 : tensor<f64>) outs(%8 : tensor<f64>) {
    ^bb0(%in: f64, %out: f64):
      %14 = arith.divf %in, %cst_0 : f64
      linalg.yield %14 : f64
    } -> tensor<f64>
    %12 = tensor.empty() : tensor<f32>
    %13 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%11 : tensor<f64>) outs(%12 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %14 = arith.truncf %in : f64 to f32
      linalg.yield %14 : f32
    } -> tensor<f32>
    return %13 : tensor<f32>
  }
}
