#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, 0, 0, 0, 0, 0)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, 0, 0, 0, 0)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, 0, 0)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4, d3, d5)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<2x5x3x4xf32>, %arg1: i64, %arg2: i64) -> tensor<2x30x4xf32> {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c5_i64 = arith.constant 5 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %0 = tensor.empty() : tensor<2xi64>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<2xi64>) {
    ^bb0(%out: i64):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i64
      linalg.yield %15 : i64
    } -> tensor<2xi64>
    %expanded = tensor.expand_shape %1 [[0, 1]] : tensor<2xi64> into tensor<1x2xi64>
    %expanded_0 = tensor.expand_shape %1 [[0, 1]] : tensor<2xi64> into tensor<2x1xi64>
    %2 = tensor.empty() : tensor<2x2xi64>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded, %expanded_0 : tensor<1x2xi64>, tensor<2x1xi64>) outs(%2 : tensor<2x2xi64>) {
    ^bb0(%in: i64, %in_5: i64, %out: i64):
      %14 = arith.addi %in, %in_5 : i64
      linalg.yield %14 : i64
    } -> tensor<2x2xi64>
    %4 = tensor.empty() : tensor<3xi64>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<3xi64>) {
    ^bb0(%out: i64):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i64
      linalg.yield %15 : i64
    } -> tensor<3xi64>
    %expanded_1 = tensor.expand_shape %5 [[0, 1]] : tensor<3xi64> into tensor<3x1xi64>
    %6 = tensor.empty() : tensor<3x2xi64>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%expanded, %expanded_1 : tensor<1x2xi64>, tensor<3x1xi64>) outs(%6 : tensor<3x2xi64>) {
    ^bb0(%in: i64, %in_5: i64, %out: i64):
      %14 = arith.addi %in, %in_5 : i64
      linalg.yield %14 : i64
    } -> tensor<3x2xi64>
    %expanded_2 = tensor.expand_shape %3 [[0], [1, 2, 3]] : tensor<2x2xi64> into tensor<2x2x1x1xi64>
    %8 = tensor.empty() : tensor<5xi64>
    %9 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%8 : tensor<5xi64>) {
    ^bb0(%out: i64):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i64
      linalg.yield %15 : i64
    } -> tensor<5xi64>
    %expanded_3 = tensor.expand_shape %9 [[0, 1, 2, 3, 4]] : tensor<5xi64> into tensor<5x1x1x1x1xi64>
    %expanded_4 = tensor.expand_shape %1 [[0, 1, 2, 3, 4, 5]] : tensor<2xi64> into tensor<2x1x1x1x1x1xi64>
    %10 = tensor.empty() : tensor<2x5x2x2x3x2xf32>
    %11 = linalg.generic {indexing_maps = [#map4, #map5, #map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4, %expanded_3, %expanded_2, %7 : tensor<2x1x1x1x1x1xi64>, tensor<5x1x1x1x1xi64>, tensor<2x2x1x1xi64>, tensor<3x2xi64>) outs(%10 : tensor<2x5x2x2x3x2xf32>) {
    ^bb0(%in: i64, %in_5: i64, %in_6: i64, %in_7: i64, %out: f32):
      %14 = arith.cmpi slt, %in, %c0_i64 : i64
      %15 = arith.addi %in, %c2_i64 : i64
      %16 = arith.select %14, %15, %in : i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = arith.cmpi slt, %in_5, %c0_i64 : i64
      %19 = arith.addi %in_5, %c5_i64 : i64
      %20 = arith.select %18, %19, %in_5 : i64
      %21 = arith.index_cast %20 : i64 to index
      %22 = arith.cmpi slt, %in_6, %c0_i64 : i64
      %23 = arith.addi %in_6, %c3_i64 : i64
      %24 = arith.select %22, %23, %in_6 : i64
      %25 = arith.index_cast %24 : i64 to index
      %26 = arith.cmpi slt, %in_7, %c0_i64 : i64
      %27 = arith.addi %in_7, %c4_i64 : i64
      %28 = arith.select %26, %27, %in_7 : i64
      %29 = arith.index_cast %28 : i64 to index
      %extracted = tensor.extract %arg0[%17, %21, %25, %29] : tensor<2x5x3x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<2x5x2x2x3x2xf32>
    %12 = tensor.empty() : tensor<2x5x2x3x2x2xf32>
    %13 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<2x5x2x2x3x2xf32>) outs(%12 : tensor<2x5x2x3x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x5x2x3x2x2xf32>
    %collapsed = tensor.collapse_shape %13 [[0], [1, 2, 3], [4, 5]] : tensor<2x5x2x3x2x2xf32> into tensor<2x30x4xf32>
    return %collapsed : tensor<2x30x4xf32>
  }
}
