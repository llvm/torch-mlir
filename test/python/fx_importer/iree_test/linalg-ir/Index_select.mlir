#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64, %arg2: tensor<4xi64>) -> tensor<4x128xf32> {
    %0 = tensor.empty() : tensor<4x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<4xi64>) outs(%0 : tensor<4x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %2 = arith.index_cast %in : i64 to index
      %3 = linalg.index 1 : index
      %extracted = tensor.extract %arg0[%2, %3] : tensor<128x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
}
