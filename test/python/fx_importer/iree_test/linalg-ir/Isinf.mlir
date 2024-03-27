#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<128x128xf32>) -> tensor<128x128xi1> {
    %cst = arith.constant 0x7F800000 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = math.absf %in : f32
      linalg.yield %4 : f32
    } -> tensor<128x128xf32>
    %2 = tensor.empty() : tensor<128x128xi1>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x128xf32>) outs(%2 : tensor<128x128xi1>) {
    ^bb0(%in: f32, %out: i1):
      %4 = arith.cmpf oeq, %in, %cst : f32
      linalg.yield %4 : i1
    } -> tensor<128x128xi1>
    return %3 : tensor<128x128xi1>
  }
}
