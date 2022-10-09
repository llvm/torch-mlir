// RUN: torch-mlir-dialects-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_broadcast(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<1x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: index) -> tensor<?x?xf32>
// CHECK:         %[[BCAST:.*]] = tcp.broadcast %[[ARG0]], %[[ARG1]] {axes = [0]} : tensor<1x?xf32>, index -> tensor<?x?xf32>
// CHECK:         return %[[BCAST]] : tensor<?x?xf32>
func.func @test_broadcast(%arg0 : tensor<1x?xf32>, %arg1 : index) -> tensor<?x?xf32> {
  %0 = "tcp.broadcast"(%arg0, %arg1) {axes = [0]} : (tensor<1x?xf32>, index) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_broadcast_multiple_dims(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x1x?x1xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: index,
// CHECK-SAME:          %[[ARG2:.*]]: index) -> tensor<?x?x?x?xf32>
// CHECK:         %[[BCAST:.*]] = tcp.broadcast %[[ARG0]], %[[ARG1]], %[[ARG2]] {axes = [1, 3]} : tensor<?x1x?x1xf32>, index, index -> tensor<?x?x?x?xf32>
// CHECK:         return %[[BCAST]] : tensor<?x?x?x?xf32>
func.func @test_broadcast_multiple_dims(%arg0 : tensor<?x1x?x1xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?xf32> {
  %0 = "tcp.broadcast"(%arg0, %arg1, %arg2) {axes = [1, 3]} : (tensor<?x1x?x1xf32>, index, index) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_broadcast_diff_rank(%arg0 : tensor<?xf32>, %arg1 : index) -> tensor<?x?xf32> {
  // expected-error@+1{{'tcp.broadcast' op failed to verify that all of {in, out} have same rank}}
  %0 = "tcp.broadcast"(%arg0, %arg1) {axes = [0]} : (tensor<?xf32>, index) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_broadcast_diff_elem_type(%arg0 : tensor<1x?xf32>, %arg1 : index) -> tensor<?x?xi32> {
  // expected-error@+1{{'tcp.broadcast' op failed to verify that all of {in, out} have same element type}}
  %0 = "tcp.broadcast"(%arg0, %arg1) {axes = [0]} : (tensor<1x?xf32>, index) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @test_broadcast_diff_num_axes(%arg0 : tensor<1x1xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  // expected-error@+1{{'tcp.broadcast' op failed to verify that argument `new_dim_sizes` has the same size as the attribute `axes`}}
  %0 = "tcp.broadcast"(%arg0, %arg1, %arg2) {axes = [0]} : (tensor<1x1xf32>, index, index) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_broadcast_axes_not_sorted(%arg0 : tensor<?x1x?x1xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?xf32> {
  // expected-error@+1{{'tcp.broadcast' op failed to verify that attribute `axes` must be in increasing order}}
  %0 = "tcp.broadcast"(%arg0, %arg1, %arg2) {axes = [3, 1]} : (tensor<?x1x?x1xf32>, index, index) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @test_broadcast_axes_w_duplicates(%arg0 : tensor<?x1x?x1xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?x?x?xf32> {
  // expected-error@+1{{'tcp.broadcast' op failed to verify that attribute `axes` must not have any duplicates}}
  %0 = "tcp.broadcast"(%arg0, %arg1, %arg2) {axes = [1, 1]} : (tensor<?x1x?x1xf32>, index, index) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
