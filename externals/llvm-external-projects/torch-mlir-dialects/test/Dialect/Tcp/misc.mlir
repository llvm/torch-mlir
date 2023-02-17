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

// -----

// CHECK-LABEL: func.func @test_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[GROUP:.*]] = tcp.group {
// CHECK:            %[[ADD:.*]] = tcp.add %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:         } : tensor<?x?xf32>
// CHECK:         return %[[GROUP]] : tensor<?x?xf32>
func.func @test_group(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) : () -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

func.func @test_group_no_yield(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{'tcp.group' op failed to verify that op region ends with a terminator}}
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
  }) : () -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

func.func @test_group_incorrect_yield_args(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{'tcp.group' op failed to verify that the number of yielded values is same as the number of results}}
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %2, %3 : tensor<?x?xf32>, tensor<?x?xf32>
  }) : () -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

func.func @test_group_incorrect_yield_arg_type(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{'tcp.group' op failed to verify that the type of operand #0 of terminator matches the corresponding result type}}
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) : () -> tensor<?xf32>
  return %10 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @test_isolated_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[IGROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]] {
// CHECK:         ^bb0(%[[BBARG0:.*]]: tensor<?x?xf32>, %[[BBARG1:.*]]: tensor<?x?xf32>):
// CHECK:            %[[ADD:.*]] = tcp.add %[[BBARG0]], %[[BBARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:         } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[IGROUP]] : tensor<?x?xf32>
func.func @test_isolated_group(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.isolated_group" (%arg0, %arg1) ({
    ^bb0(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>) :
      %2 = tcp.add %0, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

func.func @test_isolated_group_no_yield(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{'tcp.isolated_group' op failed to verify that op region ends with a terminator}}
  %10 = "tcp.isolated_group" (%arg0, %arg1) ({
    ^bb0(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>) :
      %2 = tcp.add %0, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
  }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

func.func @test_isolated_group_incorrect_yield_args(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{'tcp.isolated_group' op failed to verify that the number of yielded values is same as the number of results}}
  %10 = "tcp.isolated_group" (%arg0, %arg1) ({
    ^bb0(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>) :
      %2 = tcp.add %0, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %2, %3 : tensor<?x?xf32>, tensor<?x?xf32>
  }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

func.func @test_isolated_group_incorrect_yield_arg_type(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{'tcp.isolated_group' op failed to verify that the type of operand #0 of terminator matches the corresponding result type}}
  %10 = "tcp.isolated_group" (%arg0, %arg1) ({
    ^bb0(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>) :
      %2 = tcp.add %0, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?xf32>
  return %10 : tensor<?xf32>
}
