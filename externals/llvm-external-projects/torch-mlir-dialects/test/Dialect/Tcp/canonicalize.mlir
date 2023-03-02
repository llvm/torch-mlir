// RUN: torch-mlir-dialects-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @test_constant_folding() -> tensor<f32>
// CHECK:         %[[CONST0:.*]] = tcp.const {value = dense<2.500000e+00> : tensor<f32>} : tensor<f32>
// CHECK:         %[[MUL:.*]] = tcp.mul %[[CONST0]], %[[CONST0]] : tensor<f32>, tensor<f32> -> tensor<f32>
// CHECK:         return %[[MUL]] : tensor<f32>
func.func @test_constant_folding() -> tensor<f32> {
  %0 = tcp.const {value = dense<2.5> : tensor<f32>} : tensor<f32>
  %1 = tcp.const {value = dense<2.5> : tensor<f32>} : tensor<f32>
  %2 = tcp.mul %0, %1 : tensor<f32>, tensor<f32> -> tensor<f32>
  return %2 : tensor<f32>
}
