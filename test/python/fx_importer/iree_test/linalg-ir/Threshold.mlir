#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: f64, %arg2: f64) -> tensor<128x128xf32> {
    %cst = arith.constant 1.000000e-01 : f64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.truncf %cst : f64 to f32
      %3 = arith.cmpf ule, %in, %2 : f32
      %4 = arith.select %3, %cst_0, %in : f32
      linalg.yield %4 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
