#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<3x2xf32>, %arg1: i64, %arg2: tensor<3x2xi64>) -> tensor<3x2xf32> {
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<3x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x2xf32>) -> tensor<3x2xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<3x2xi64>) outs(%1 : tensor<3x2xf32>) {
    ^bb0(%in: i64, %out: f32):
      %3 = linalg.index 0 : index
      %4 = arith.index_cast %in : i64 to index
      %5 = arith.cmpi slt, %4, %c2 : index
      cf.assert %5, "index must be smaller than dim size"
      %6 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %6, "index must be larger or equal to 0"
      %extracted = tensor.extract %arg0[%3, %4] : tensor<3x2xf32>
      linalg.yield %extracted : f32
    } -> tensor<3x2xf32>
    return %2 : tensor<3x2xf32>
  }
}
