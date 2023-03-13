// RUN: torch-mlir-dialects-opt <%s -convert-stablehlo-to-tcp -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tanh(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:         %[[TANH:.*]] = tcp.tanh %[[ARG0]] : tensor<f32>
// CHECK:         return %[[TANH]] : tensor<f32>
// CHECK:       }
func.func @tanh(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.tanh %arg0 : tensor<f32>
  return %0 : tensor<f32>
}
