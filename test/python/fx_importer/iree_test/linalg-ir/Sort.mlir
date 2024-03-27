#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xi64>) {
    %0 = tensor.empty() : tensor<128x128xi64>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<128x128xi64>) {
    ^bb0(%out: i64):
      %3 = linalg.index 1 : index
      %4 = arith.index_cast %3 : index to i64
      linalg.yield %4 : i64
    } -> tensor<128x128xi64>
    %2:2 = tm_tensor.sort dimension(1) outs(%arg0, %1 : tensor<128x128xf32>, tensor<128x128xi64>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: i64, %arg4: i64):
      %3 = arith.cmpf ole, %arg1, %arg2 : f32
      tm_tensor.yield %3 : i1
    } -> tensor<128x128xf32>, tensor<128x128xi64>
    return %2#0, %2#1 : tensor<128x128xf32>, tensor<128x128xi64>
  }
}
