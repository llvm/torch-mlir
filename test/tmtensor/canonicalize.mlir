// RUN: torch-mlir-dialects-opt -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @tensor.cast(
func.func @tensor.cast(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  %init = tensor.empty() : tensor<128xi32>
  %c0 = tensor.empty() : tensor<i32>

  %casted_arg0 = tensor.cast %arg0 : tensor<128xi32> to tensor<?xi32>
  %casted_init = tensor.cast %init : tensor<128xi32> to tensor<?xi32>
 // CHECK:      tm_tensor.scan
 // CHECK-SAME:   ins(%{{[a-zA-Z0-9]*}} : tensor<128xi32>)
 // CHECK-SAME:  outs(%{{[a-zA-Z0-9]*}}, %{{[a-zA-Z0-9]*}} : tensor<128xi32>, tensor<i32>)
  %0, %1 = tm_tensor.scan dimension(0) inclusive(true)
       ins(%casted_arg0 : tensor<?xi32>)
       outs(%casted_init, %c0: tensor<?xi32>, tensor<i32>) {
       ^bb0(%barg0 : i32, %barg1 : i32, %barg2 : i32):
         %sum = arith.addi %barg0, %barg1 : i32
         tm_tensor.yield %sum : i32
  } -> tensor<?xi32>, tensor<i32>

  %2 = tensor.cast %0: tensor<?xi32> to tensor<128xi32>

  return %2: tensor<128xi32>
}
