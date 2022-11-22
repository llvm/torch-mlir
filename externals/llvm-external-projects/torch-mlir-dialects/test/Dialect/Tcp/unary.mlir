// RUN: torch-mlir-dialects-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_tanh_f32(
// CHECK-SAME:               %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[TANH:.*]] = tcp.tanh %[[ARG]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[TANH]] : tensor<?x?xf32>
func.func @test_tanh_f32(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_tanh_diff_rank(%arg0 : tensor<?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1{{'tcp.tanh' op all non-scalar operands/results must have the same shape and base type}}
  %0 = tcp.tanh %arg0 : tensor<?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_tanh_diff_shape(%arg0 : tensor<5x?xf32>) -> tensor<6x?xf32> {
  // expected-error@+1{{'tcp.tanh' op all non-scalar operands/results must have the same shape and base type}}
  %0 = tcp.tanh %arg0 : tensor<5x?xf32> -> tensor<6x?xf32>
  return %0 : tensor<6x?xf32>
}
