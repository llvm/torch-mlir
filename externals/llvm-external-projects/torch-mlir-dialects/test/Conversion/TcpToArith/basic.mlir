// RUN: torch-mlir-dialects-opt <%s -convert-tcp-to-arith -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_constants() -> tensor<f32> {
// CHECK:         %[[C0:.*]] = arith.constant dense<2.500000e+00> : tensor<f32>
// CHECK:         %[[C1:.*]] = arith.constant dense<[3, 6, 10]> : tensor<3xi32>
// CHECK:         %[[C2:.*]] = arith.constant
// CHECK-SAME{LITERAL}:   dense<[[2, 3, 5], [20, 25, 30]]> : tensor<2x3xi64>
// CHECK:         return %[[C0]] : tensor<f32>
// CHECK:       }
func.func @test_constants() -> tensor<f32> {
  %0 = "tcp.const"() {value = dense<2.5> : tensor<f32>} : () -> tensor<f32>
  %1 = "tcp.const"() {value = dense<[3, 6, 10]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tcp.const"() {value = dense<[[2, 3, 5], [20, 25, 30]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
  return %0 : tensor<f32>
}
