#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x64xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [128, 64] [1, 1] : tensor<128x128xf32> to tensor<128x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, 64] [128, 64] [1, 1] : tensor<128x128xf32> to tensor<128x64xf32>
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_0 : tensor<128x64xf32>) outs(%0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.negf %in : f32
      %4 = math.exp %3 : f32
      %5 = arith.addf %4, %cst : f32
      %6 = arith.divf %cst, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<128x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %1 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%0 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<128x64xf32>
    return %2 : tensor<128x64xf32>
  }
}
