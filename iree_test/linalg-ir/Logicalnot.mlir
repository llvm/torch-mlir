#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xi1> {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty() : tensor<128x128xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %2 = arith.extf %in : f32 to f64
      %3 = arith.cmpf oeq, %2, %cst : f64
      linalg.yield %3 : i1
    } -> tensor<128x128xi1>
    return %1 : tensor<128x128xi1>
  }
}
