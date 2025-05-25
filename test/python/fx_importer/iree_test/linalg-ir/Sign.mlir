#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0, d1) -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_2 = arith.constant dense<-1.000000e+00> : tensor<f64>
    %0 = tensor.empty() : tensor<128x128xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %10 = arith.cmpf ugt, %in, %cst_0 : f32
      linalg.yield %10 : i1
    } -> tensor<128x128xi1>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %10 = arith.cmpf uge, %in, %cst_0 : f32
      linalg.yield %10 : i1
    } -> tensor<128x128xi1>
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = []} ins(%cst : tensor<f64>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %10 = arith.truncf %in : f64 to f32
      linalg.yield %10 : f32
    } -> tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = []} ins(%cst_1 : tensor<f64>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %10 = arith.truncf %in : f64 to f32
      linalg.yield %10 : f32
    } -> tensor<f32>
    %6 = tensor.empty() : tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map2, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %4, %5 : tensor<128x128xi1>, tensor<f32>, tensor<f32>) outs(%6 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %in_3: f32, %in_4: f32, %out: f32):
      %10 = arith.select %in, %in_3, %in_4 : f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = []} ins(%cst_2 : tensor<f64>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %10 = arith.truncf %in : f64 to f32
      linalg.yield %10 : f32
    } -> tensor<f32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %7, %8 : tensor<128x128xi1>, tensor<128x128xf32>, tensor<f32>) outs(%6 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %in_3: f32, %in_4: f32, %out: f32):
      %10 = arith.select %in, %in_3, %in_4 : f32
      linalg.yield %10 : f32
    } -> tensor<128x128xf32>
    return %9 : tensor<128x128xf32>
  }
}
