#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -5.000000e-01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0.398942292 : f32
    %cst_3 = arith.constant 1.41421354 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %2 = arith.mulf %in_4, %in_4 : f32
      %3 = arith.mulf %2, %cst_0 : f32
      %4 = math.exp %3 : f32
      %5 = arith.divf %in_4, %cst_3 : f32
      %6 = math.erf %5 : f32
      %7 = arith.addf %6, %cst_1 : f32
      %8 = arith.mulf %7, %cst : f32
      %9 = arith.mulf %4, %in_4 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.addf %10, %8 : f32
      %12 = arith.mulf %in, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
