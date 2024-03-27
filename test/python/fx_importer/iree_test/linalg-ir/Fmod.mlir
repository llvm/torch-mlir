#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.divf %in, %in_0 : f32
      %3 = math.trunc %2 : f32
      %4 = arith.mulf %3, %in_0 : f32
      %5 = arith.subf %in, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
}
