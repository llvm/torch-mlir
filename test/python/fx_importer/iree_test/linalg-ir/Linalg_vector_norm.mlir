#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<() -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<f32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%arg0 : tensor<128x128xf32>) outs(%1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = math.absf %in : f32
      %5 = math.powf %4, %cst_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%2 : tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = math.powf %in, %cst_1 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
    return %3 : tensor<f32>
  }
}
