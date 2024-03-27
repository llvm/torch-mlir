#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<f32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<128xf32>) {
    ^bb0(%out: f32):
      %5 = linalg.index 0 : index
      %extracted = tensor.extract %arg0[%5, %5] : tensor<128x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<128xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%1 : tensor<128xf32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<f32>
    return %4 : tensor<f32>
  }
}
