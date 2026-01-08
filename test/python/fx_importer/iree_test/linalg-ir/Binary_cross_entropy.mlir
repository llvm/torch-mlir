#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0, d1) -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<f32> {
    %cst = arith.constant dense<-100> : tensor<i64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 1.638400e+04 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.subf %in, %cst_1 : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.negf %in : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = math.log1p %in : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %4 = tensor.empty() : tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = []} ins(%cst : tensor<i64>) outs(%4 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %15 = arith.sitofp %in : i64 to f32
      linalg.yield %15 : f32
    } -> tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %5 : tensor<128x128xf32>, tensor<f32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.cmpf ogt, %in, %in_3 : f32
      %16 = arith.select %15, %in, %in_3 : f32
      linalg.yield %16 : f32
    } -> tensor<128x128xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %6 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.mulf %in, %in_3 : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = math.log %in : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %5 : tensor<128x128xf32>, tensor<f32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.cmpf ogt, %in, %in_3 : f32
      %16 = arith.select %15, %in, %in_3 : f32
      linalg.yield %16 : f32
    } -> tensor<128x128xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %9 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.mulf %in, %in_3 : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.subf %in, %in_3 : f32
      linalg.yield %15 : f32
    } -> tensor<128x128xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<f32>) -> tensor<f32>
    %13 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "reduction"]} ins(%11 : tensor<128x128xf32>) outs(%12 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.addf %in, %out : f32
      linalg.yield %15 : f32
    } -> tensor<f32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = []} ins(%13 : tensor<f32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.divf %in, %cst_2 : f32
      linalg.yield %15 : f32
    } -> tensor<f32>
    return %14 : tensor<f32>
  }
}
