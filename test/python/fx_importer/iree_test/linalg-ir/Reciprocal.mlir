#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf one, %in, %cst : f32
      cf.assert %2, "unimplemented: tensor with zero element"
      %3 = arith.divf %cst_0, %in : f32
      linalg.yield %3 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
