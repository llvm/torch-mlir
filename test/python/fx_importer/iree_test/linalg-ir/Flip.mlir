#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c127 = arith.constant 127 : index
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%1 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = arith.subi %c127, %4 : index
      %extracted = tensor.extract %arg0[%3, %5] : tensor<128x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
}
