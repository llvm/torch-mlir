#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<() -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<3x5xf32>, %arg1: tensor<3xi64>) -> tensor<f32> {
    %cst = arith.constant dense<0> : tensor<i64>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c5 = arith.constant 5 : index
    %c0_i64 = arith.constant 0 : i64
    %c-100_i64 = arith.constant -100 : i64
    %0 = tensor.empty() : tensor<3xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : tensor<3xi64>) outs(%0 : tensor<3xi1>) {
    ^bb0(%in: i64, %out: i1):
      %19 = arith.cmpi ne, %in, %c-100_i64 : i64
      linalg.yield %19 : i1
    } -> tensor<3xi1>
    %2 = tensor.empty() : tensor<3xi64>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel"]} ins(%1, %arg1, %cst : tensor<3xi1>, tensor<3xi64>, tensor<i64>) outs(%2 : tensor<3xi64>) {
    ^bb0(%in: i1, %in_1: i64, %in_2: i64, %out: i64):
      %19 = arith.select %in, %in_1, %in_2 : i64
      linalg.yield %19 : i64
    } -> tensor<3xi64>
    %expanded = tensor.expand_shape %3 [[0, 1]] : tensor<3xi64> into tensor<3x1xi64>
    %4 = tensor.empty() : tensor<3x1xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<3x1xf32>) -> tensor<3x1xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<3x1xi64>) outs(%5 : tensor<3x1xf32>) {
    ^bb0(%in: i64, %out: f32):
      %19 = linalg.index 0 : index
      %20 = arith.index_cast %in : i64 to index
      %21 = arith.cmpi slt, %20, %c5 : index
      cf.assert %21, "index must be smaller than dim size"
      %22 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %22, "index must be larger or equal to 0"
      %extracted = tensor.extract %arg0[%19, %20] : tensor<3x5xf32>
      linalg.yield %extracted : f32
    } -> tensor<3x1xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1]] : tensor<3x1xf32> into tensor<3xf32>
    %7 = tensor.empty() : tensor<3xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%collapsed : tensor<3xf32>) outs(%7 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.negf %in : f32
      linalg.yield %19 : f32
    } -> tensor<3xf32>
    %9 = tensor.empty() : tensor<f32>
    %10 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%cst : tensor<i64>) outs(%9 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %19 = arith.sitofp %in : i64 to f32
      linalg.yield %19 : f32
    } -> tensor<f32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel"]} ins(%1, %8, %10 : tensor<3xi1>, tensor<3xf32>, tensor<f32>) outs(%7 : tensor<3xf32>) {
    ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
      %19 = arith.select %in, %in_1, %in_2 : f32
      linalg.yield %19 : f32
    } -> tensor<3xf32>
    %12 = tensor.empty() : tensor<i64>
    %13 = linalg.fill ins(%c0_i64 : i64) outs(%12 : tensor<i64>) -> tensor<i64>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%1 : tensor<3xi1>) outs(%13 : tensor<i64>) {
    ^bb0(%in: i1, %out: i64):
      %19 = arith.extui %in : i1 to i64
      %20 = arith.addi %19, %out : i64
      linalg.yield %20 : i64
    } -> tensor<i64>
    %15 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%14 : tensor<i64>) outs(%9 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %19 = arith.sitofp %in : i64 to f32
      linalg.yield %19 : f32
    } -> tensor<f32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<f32>) -> tensor<f32>
    %17 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%11 : tensor<3xf32>) outs(%16 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.addf %in, %out : f32
      linalg.yield %19 : f32
    } -> tensor<f32>
    %18 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = []} ins(%17, %15 : tensor<f32>, tensor<f32>) outs(%9 : tensor<f32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %19 = arith.divf %in, %in_1 : f32
      linalg.yield %19 : f32
    } -> tensor<f32>
    return %18 : tensor<f32>
  }
}
