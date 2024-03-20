#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: f64) -> tensor<128x128xi1> {
    %cst = arith.constant 5.000000e-01 : f32
    %0 = tensor.empty() : tensor<128x128xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %2 = arith.cmpf ult, %in, %cst : f32
      linalg.yield %2 : i1
    } -> tensor<128x128xi1>
    return %1 : tensor<128x128xi1>
  }
}