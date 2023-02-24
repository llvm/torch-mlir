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

// -----

// CHECK-LABEL: func.func @test_clamp_f32(
// CHECK-SAME:               %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[CLAMP:.*]] = tcp.clamp %[[ARG]] {min_float = 0.000000e+00 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[CLAMP]] : tensor<?x?xf32>
func.func @test_clamp_f32(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.clamp %arg0 { min_float = 0.0 : f32 } : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_clamp_i32(
// CHECK-SAME:               %[[ARG:.*]]: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:         %[[CLAMP:.*]] = tcp.clamp %[[ARG]] {max_int = 6 : i64} : tensor<?x?xi32> -> tensor<?x?xi32>
// CHECK:         return %[[CLAMP]] : tensor<?x?xi32>
func.func @test_clamp_i32(%arg0 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = tcp.clamp %arg0 { max_int = 6 : i64 } : tensor<?x?xi32> -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @test_clamp_no_min_max(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1{{'tcp.clamp' op failed to verify that at least one of min / max attributes must be set}}
  %0 = tcp.clamp %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_clamp_min_invalid(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1{{'tcp.clamp' op failed to verify that int min / max attributes must not be set when input is a float tensor}}
  %0 = tcp.clamp %arg0 { max_int = 6 : i64 } : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_sigmoid_f32(
// CHECK-SAME:               %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[TANH:.*]] = tcp.sigmoid %[[ARG]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[TANH]] : tensor<?x?xf32>
func.func @test_sigmoid_f32(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sigmoid %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_sqrt_f32(
// CHECK-SAME:               %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[SQRT:.*]] = tcp.sqrt %[[ARG]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[SQRT]] : tensor<?x?xf32>
func.func @test_sqrt_f32(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sqrt %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
