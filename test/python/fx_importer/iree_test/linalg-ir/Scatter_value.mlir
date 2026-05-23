#map = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0, 0)>
#map4 = affine_map<(d0) -> (d0)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64, %arg2: tensor<128x128xi64>, %arg3: i64) -> tensor<128x128xf32> {
    %cst = arith.constant dense<5> : tensor<i64>
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%cst : tensor<i64>) outs(%0 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %12 = arith.sitofp %in : i64 to f32
      linalg.yield %12 : f32
    } -> tensor<f32>
    %2 = tensor.empty() : tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<f32>) outs(%2 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x128xf32>
    %4 = tensor.empty() : tensor<16384x1xi32>
    %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<16384x1xi32>) -> tensor<16384x1xi32>
    %6 = tensor.empty() : tensor<16384xf32>
    %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<16384xf32>) -> tensor<16384xf32>
    %8:3 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel"]} outs(%5, %5, %7 : tensor<16384x1xi32>, tensor<16384x1xi32>, tensor<16384xf32>) {
    ^bb0(%out: i32, %out_2: i32, %out_3: f32):
      %12 = linalg.index 0 : index
      %13 = arith.remsi %12, %c128 : index
      %14 = arith.divsi %12, %c128 : index
      %15 = arith.remsi %14, %c128 : index
      %extracted = tensor.extract %arg2[%15, %13] : tensor<128x128xi64>
      %extracted_4 = tensor.extract %3[%15, %13] : tensor<128x128xf32>
      %16 = arith.index_cast %15 : index to i64
      %17 = arith.trunci %16 : i64 to i32
      %18 = arith.trunci %extracted : i64 to i32
      linalg.yield %17, %18, %extracted_4 : i32, i32, f32
    } -> (tensor<16384x1xi32>, tensor<16384x1xi32>, tensor<16384xf32>)
    %9 = tensor.empty() : tensor<16384x2xi32>
    %10 = linalg.fill ins(%c0_i32 : i32) outs(%9 : tensor<16384x2xi32>) -> tensor<16384x2xi32>
    %inserted_slice = tensor.insert_slice %8#0 into %10[0, 0] [16384, 1] [1, 1] : tensor<16384x1xi32> into tensor<16384x2xi32>
    %inserted_slice_1 = tensor.insert_slice %8#1 into %inserted_slice[0, 1] [16384, 1] [1, 1] : tensor<16384x1xi32> into tensor<16384x2xi32>
    %11 = tm_tensor.scatter {dimension_map = array<i64: 0, 1>} unique_indices(false) ins(%8#2, %inserted_slice_1 : tensor<16384xf32>, tensor<16384x2xi32>) outs(%arg0 : tensor<128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      tm_tensor.yield %arg4 : f32
    } -> tensor<128x128xf32>
    return %11 : tensor<128x128xf32>
  }
}
