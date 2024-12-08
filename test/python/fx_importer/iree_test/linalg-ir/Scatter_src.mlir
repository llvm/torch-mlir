#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64, %arg2: tensor<128x128xi64>, %arg3: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16384x1xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<16384x1xi32>) -> tensor<16384x1xi32>
    %2 = tensor.empty() : tensor<16384xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16384xf32>) -> tensor<16384xf32>
    %4:3 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel"]} outs(%1, %1, %3 : tensor<16384x1xi32>, tensor<16384x1xi32>, tensor<16384xf32>) {
    ^bb0(%out: i32, %out_1: i32, %out_2: f32):
      %8 = linalg.index 0 : index
      %9 = arith.remsi %8, %c128 : index
      %10 = arith.divsi %8, %c128 : index
      %11 = arith.remsi %10, %c128 : index
      %extracted = tensor.extract %arg2[%11, %9] : tensor<128x128xi64>
      %extracted_3 = tensor.extract %arg3[%11, %9] : tensor<128x128xf32>
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.trunci %12 : i64 to i32
      %14 = arith.trunci %extracted : i64 to i32
      linalg.yield %13, %14, %extracted_3 : i32, i32, f32
    } -> (tensor<16384x1xi32>, tensor<16384x1xi32>, tensor<16384xf32>)
    %5 = tensor.empty() : tensor<16384x2xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16384x2xi32>) -> tensor<16384x2xi32>
    %inserted_slice = tensor.insert_slice %4#0 into %6[0, 0] [16384, 1] [1, 1] : tensor<16384x1xi32> into tensor<16384x2xi32>
    %inserted_slice_0 = tensor.insert_slice %4#1 into %inserted_slice[0, 1] [16384, 1] [1, 1] : tensor<16384x1xi32> into tensor<16384x2xi32>
    %7 = tm_tensor.scatter {dimension_map = array<i64: 0, 1>} unique_indices(false) ins(%4#2, %inserted_slice_0 : tensor<16384xf32>, tensor<16384x2xi32>) outs(%arg0 : tensor<128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      tm_tensor.yield %arg4 : f32
    } -> tensor<128x128xf32>
    return %7 : tensor<128x128xf32>
  }
}
