#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.matmul ins(%arg1, %arg2 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg0 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<128x128xf32>
    return %3 : tensor<128x128xf32>
  }
}
