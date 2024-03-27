module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64) -> tensor<128x2xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [128, 2] [1, 4] : tensor<128x128xf32> to tensor<128x2xf32>
    return %extracted_slice : tensor<128x2xf32>
  }
}
