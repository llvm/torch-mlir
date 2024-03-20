module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<1024xf32>, %arg1: i64) -> tensor<1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
    %4:2 = tm_tensor.scan dimension(0) inclusive(true) ins(%arg0 : tensor<1024xf32>) outs(%1, %3 : tensor<1024xf32>, tensor<f32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %5 = arith.addf %arg2, %arg3 : f32
      tm_tensor.yield %5 : f32
    } -> tensor<1024xf32>, tensor<f32>
    return %4#0 : tensor<1024xf32>
  }
}
