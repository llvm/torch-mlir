#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> tensor<128x128xi64> {
    %c2_i64 = arith.constant 2 : i64
    %0 = tensor.empty() : tensor<128x128xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<128x128xi64>) outs(%0 : tensor<128x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %c2_i64 : i64
    } -> tensor<128x128xi64>
    return %1 : tensor<128x128xi64>
  }
}
