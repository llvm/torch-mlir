#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant -5.000000e-01 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ult, %in, %cst : f32
      %3 = arith.select %2, %cst, %in : f32
      %4 = arith.cmpf ugt, %3, %cst_0 : f32
      %5 = arith.select %4, %cst_0, %3 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
