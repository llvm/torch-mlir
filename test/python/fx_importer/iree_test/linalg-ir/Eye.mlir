#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1) -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: i64) -> tensor<128x128xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f64>
    %0 = tensor.empty() : tensor<128xi64>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<128xi64>) {
    ^bb0(%out: i64):
      %9 = linalg.index 0 : index
      %10 = arith.index_cast %9 : index to i64
      linalg.yield %10 : i64
    } -> tensor<128xi64>
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    %2 = tensor.empty() : tensor<128x128xi1>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded, %1 : tensor<128x1xi64>, tensor<128xi64>) outs(%2 : tensor<128x128xi1>) {
    ^bb0(%in: i64, %in_1: i64, %out: i1):
      %9 = arith.cmpi eq, %in, %in_1 : i64
      linalg.yield %9 : i1
    } -> tensor<128x128xi1>
    %4 = tensor.empty() : tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%cst_0 : tensor<f64>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %9 = arith.truncf %in : f64 to f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%cst : tensor<f64>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %9 = arith.truncf %in : f64 to f32
      linalg.yield %9 : f32
    } -> tensor<f32>
    %7 = tensor.empty() : tensor<128x128xf32>
    %8 = linalg.generic {indexing_maps = [#map3, #map5, #map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%3, %5, %6 : tensor<128x128xi1>, tensor<f32>, tensor<f32>) outs(%7 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
      %9 = arith.select %in, %in_1, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<128x128xf32>
    return %8 : tensor<128x128xf32>
  }
}
