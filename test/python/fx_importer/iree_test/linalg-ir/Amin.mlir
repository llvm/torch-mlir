#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>, %arg1: i64) -> tensor<128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0x7F800000 : f32
    %0 = tensor.empty() : tensor<128xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32>
    %2 = tensor.empty() : tensor<128xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%3, %1 : tensor<128xf32>, tensor<128xi32>) {
    ^bb0(%in: f32, %out: f32, %out_0: i32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.minimumf %in, %out : f32
      %8 = arith.cmpf olt, %in, %out : f32
      %9 = arith.select %8, %6, %out_0 : i32
      linalg.yield %7, %9 : f32, i32
    } -> (tensor<128xf32>, tensor<128xi32>)
    return %4#0 : tensor<128xf32>
  }
}
