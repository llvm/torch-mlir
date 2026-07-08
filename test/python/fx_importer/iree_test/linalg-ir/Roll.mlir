module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64) -> tensor<128x128xf32> {
    %extracted_slice = tensor.extract_slice %arg0[126, 0] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
    %cast = tensor.cast %extracted_slice : tensor<2x128xf32> to tensor<?x128xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[0, 0] [126, 128] [1, 1] : tensor<128x128xf32> to tensor<126x128xf32>
    %cast_1 = tensor.cast %extracted_slice_0 : tensor<126x128xf32> to tensor<?x128xf32>
    %concat = tensor.concat dim(0) %cast, %cast_1 : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<128x128xf32>
    %extracted_slice_2 = tensor.extract_slice %concat[0, 127] [128, 1] [1, 1] : tensor<128x128xf32> to tensor<128x1xf32>
    %cast_3 = tensor.cast %extracted_slice_2 : tensor<128x1xf32> to tensor<128x?xf32>
    %extracted_slice_4 = tensor.extract_slice %concat[0, 0] [128, 127] [1, 1] : tensor<128x128xf32> to tensor<128x127xf32>
    %cast_5 = tensor.cast %extracted_slice_4 : tensor<128x127xf32> to tensor<128x?xf32>
    %concat_6 = tensor.concat dim(1) %cast_3, %cast_5 : (tensor<128x?xf32>, tensor<128x?xf32>) -> tensor<128x128xf32>
    return %concat_6 : tensor<128x128xf32>
  }
}
