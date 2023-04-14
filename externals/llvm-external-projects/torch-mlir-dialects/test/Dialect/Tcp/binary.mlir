// RUN: torch-mlir-dialects-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_add_f32(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[ADD:.*]] = tcp.add %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[ADD]] : tensor<?x?xf32>
func.func @test_add_f32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_add_i32(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?xi32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK:         %[[ADD:.*]] = tcp.add %[[ARG0]], %[[ARG1]] : tensor<?xi32>, tensor<?xi32> -> tensor<?xi32>
// CHECK:         return %[[ADD]] : tensor<?xi32>
func.func @test_add_i32(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?xi32>, tensor<?xi32> -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func.func @test_add_diff_elem_type(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
  // expected-error@+1 {{'tcp.add' op requires the same element type for all operands and results}}
  %0 = tcp.add %arg0, %arg1 : tensor<?xf32>, tensor<?xi32> -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func.func @test_add_diff_rank(%arg0 : tensor<?x?xi32>, %arg1 : tensor<?xi32>) -> tensor<?x?xi32> {
  // expected-error@+1 {{'tcp.add' op all non-scalar operands/results must have the same shape and base type}}
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xi32>, tensor<?xi32> -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @test_add_diff_shape(%arg0 : tensor<5xi32>, %arg1 : tensor<6xi32>) -> tensor<6xi32> {
  // expected-error@+1 {{'tcp.add' op all non-scalar operands/results must have the same shape and base type}}
  %0 = tcp.add %arg0, %arg1 : tensor<5xi32>, tensor<6xi32> -> tensor<6xi32>
  return %0 : tensor<6xi32>
}

// -----

// CHECK-LABEL: func.func @test_sub(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[SUB:.*]] = tcp.sub %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[SUB]] : tensor<?x?xf32>
func.func @test_sub(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sub %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_mul(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[MUL:.*]] = tcp.mul %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[MUL]] : tensor<?x?xf32>
func.func @test_mul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.mul %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_divf(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[DIV:.*]] = tcp.divf %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[DIV]] : tensor<?x?xf32>
func.func @test_divf(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.divf %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @test_atan2_f32(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[ATAN2:.*]] = tcp.atan2 %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[ATAN2]] : tensor<?x?xf32>
func.func @test_atan2_f32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.atan2 %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}