#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64) -> (tensor<128x5xf32>, tensor<128x5xi64>) {
    %0 = tensor.empty() : tensor<128x128xi64>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<128x128xi64>) {
    ^bb0(%out: i64):
      %3 = linalg.index 1 : index
      %4 = arith.index_cast %3 : index to i64
      linalg.yield %4 : i64
    } -> tensor<128x128xi64>
    %2:2 = tm_tensor.sort dimension(1) outs(%arg0, %1 : tensor<128x128xf32>, tensor<128x128xi64>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i64, %arg5: i64):
      %3 = arith.cmpf oge, %arg2, %arg3 : f32
      tm_tensor.yield %3 : i1
    } -> tensor<128x128xf32>, tensor<128x128xi64>
    %extracted_slice = tensor.extract_slice %2#0[0, 0] [128, 5] [1, 1] : tensor<128x128xf32> to tensor<128x5xf32>
    %extracted_slice_0 = tensor.extract_slice %2#1[0, 0] [128, 5] [1, 1] : tensor<128x128xi64> to tensor<128x5xi64>
    return %extracted_slice, %extracted_slice_0 : tensor<128x5xf32>, tensor<128x5xi64>
  }
}
