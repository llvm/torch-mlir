#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<256x128xf32>, %arg1: i64) -> tensor<256x128xf32> {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-05 : f64
    %cst_2 = arith.constant 1.280000e+02 : f64
    %cst_3 = arith.constant 1.280000e+02 : f32
    %0 = tensor.empty() : tensor<256x128xf64>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<256x128xf32>) outs(%0 : tensor<256x128xf64>) {
    ^bb0(%in: f32, %out: f64):
      %20 = arith.extf %in : f32 to f64
      linalg.yield %20 : f64
    } -> tensor<256x128xf64>
    %2 = tensor.empty() : tensor<256x1xf64>
    %3 = linalg.fill ins(%cst : f64) outs(%2 : tensor<256x1xf64>) -> tensor<256x1xf64>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<256x128xf64>) outs(%3 : tensor<256x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %20 = arith.addf %in, %out : f64
      linalg.yield %20 : f64
    } -> tensor<256x1xf64>
    %5 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<256x1xf64>) outs(%2 : tensor<256x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %20 = arith.divf %in, %cst_2 : f64
      linalg.yield %20 : f64
    } -> tensor<256x1xf64>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %5 : tensor<256x128xf64>, tensor<256x1xf64>) outs(%0 : tensor<256x128xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %20 = arith.subf %in, %in_4 : f64
      linalg.yield %20 : f64
    } -> tensor<256x128xf64>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %6 : tensor<256x128xf64>, tensor<256x128xf64>) outs(%0 : tensor<256x128xf64>) {
    ^bb0(%in: f64, %in_4: f64, %out: f64):
      %20 = arith.mulf %in, %in_4 : f64
      linalg.yield %20 : f64
    } -> tensor<256x128xf64>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<256x128xf64>) outs(%3 : tensor<256x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %20 = arith.addf %in, %out : f64
      linalg.yield %20 : f64
    } -> tensor<256x1xf64>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<256x1xf64>) outs(%2 : tensor<256x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %20 = arith.divf %in, %cst_2 : f64
      linalg.yield %20 : f64
    } -> tensor<256x1xf64>
    %10 = tensor.empty() : tensor<256x1xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<256x1xf64>) outs(%10 : tensor<256x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %20 = arith.truncf %in : f64 to f32
      linalg.yield %20 : f32
    } -> tensor<256x1xf32>
    %12 = linalg.fill ins(%cst_0 : f32) outs(%10 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<256x128xf32>) outs(%12 : tensor<256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.addf %in, %out : f32
      linalg.yield %20 : f32
    } -> tensor<256x1xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<256x1xf32>) outs(%10 : tensor<256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.divf %in, %cst_3 : f32
      linalg.yield %20 : f32
    } -> tensor<256x1xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<256x1xf32>) outs(%10 : tensor<256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.truncf %cst_1 : f64 to f32
      %21 = arith.addf %in, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<256x1xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<256x1xf32>) outs(%10 : tensor<256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %20 = math.rsqrt %in : f32
      linalg.yield %20 : f32
    } -> tensor<256x1xf32>
    %17 = tensor.empty() : tensor<256x128xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %14 : tensor<256x128xf32>, tensor<256x1xf32>) outs(%17 : tensor<256x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %20 = arith.subf %in, %in_4 : f32
      linalg.yield %20 : f32
    } -> tensor<256x128xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %16 : tensor<256x128xf32>, tensor<256x1xf32>) outs(%17 : tensor<256x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %20 = arith.mulf %in, %in_4 : f32
      linalg.yield %20 : f32
    } -> tensor<256x128xf32>
    return %19 : tensor<256x128xf32>
  }
}
