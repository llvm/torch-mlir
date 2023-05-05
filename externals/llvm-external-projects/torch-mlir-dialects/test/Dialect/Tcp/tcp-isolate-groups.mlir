// RUN: torch-mlir-dialects-opt %s -split-input-file -tcp-isolate-group-ops -verify-diagnostics --mlir-print-ir-after-all | FileCheck %s

// CHECK-LABEL: func.func @test_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]] {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<?x?xf32>, %[[ARG3:.*]]: tensor<?x?xf32>):
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[ARG3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:          } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?xf32>
// CHECK:        }
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

// CHECK-LABEL: func.func @test_bigger_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG2:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]], %[[ARG2]] {
// CHECK:            ^bb0(%[[ARG3:.*]]: tensor<?x?xf32>, %[[ARG4:.*]]: tensor<?x?xf32>, %[[ARG5:.*]]: tensor<?x?xf32>):
// CHECK:              %[[CLAMP:.*]] = tcp.clamp %[[ARG3]] {min_float = 0.000000e+00 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[SUB:.*]] = tcp.sub %[[ARG4]], %[[CLAMP]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[TANH:.*]] = tcp.tanh %[[ARG5]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[ADD:.*]] = tcp.add %[[TANH]], %[[SUB]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              tcp.yield %[[ADD]] : tensor<?x?xf32>
// CHECK:          } : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?xf32>
// CHECK:        }
func.func @test_bigger_group(%arg0 : tensor<?x?xf32>,
                             %arg1 : tensor<?x?xf32>,
                             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %4 = tcp.clamp %arg0 {min_float = 0.0 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
      %5 = tcp.sub %arg1, %4 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %6 = tcp.tanh %arg2 : tensor<?x?xf32> -> tensor<?x?xf32>
      %7 = tcp.add %6, %5 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %7 : tensor<?x?xf32>
  }) : () -> tensor<?x?xf32>
   return %10 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_propagate_attrs(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]] attributes {test_attr} {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<?x?xf32>, %[[ARG3:.*]]: tensor<?x?xf32>):
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[ARG3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:          } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?xf32>
// CHECK:        }
func.func @test_propagate_attrs(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) { "test_attr" } : () -> tensor<?x?xf32>
   return %10 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_const(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<5xi32>) -> tensor<5xi32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]] {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<5xi32>):
// CHECK:              %[[CONST:.*]] = tcp.const
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[CONST]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              %[[MUL:.*]] = tcp.mul %[[ADD]], %[[CONST]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              tcp.yield %[[MUL]] : tensor<5xi32>
// CHECK:          } : tensor<5xi32> -> tensor<5xi32>
// CHECK:          return %[[GROUP]] : tensor<5xi32>
// CHECK:        }
func.func @test_const(%arg0 : tensor<5xi32>) -> tensor<5xi32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %1 = tcp.const {value = dense<[3, 6, 10, 15, 23]> : tensor<5xi32>} : tensor<5xi32>
      %2 = tcp.add %arg0, %1 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      %3 = tcp.mul %2, %1 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      tcp.yield %3 : tensor<5xi32>
  }) : () -> tensor<5xi32>
   return %10 : tensor<5xi32>
}

// -----

// CHECK-LABEL: func.func @test_inputs_with_multiple_uses(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<5xi32>) -> tensor<5xi32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]] {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<5xi32>):
// CHECK:              %[[CONST:.*]] = tcp.const
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[CONST]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              %[[MUL:.*]] = tcp.mul %[[ADD]], %[[ARG2]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              tcp.yield %[[MUL]] : tensor<5xi32>
// CHECK:          } : tensor<5xi32> -> tensor<5xi32>
// CHECK:          return %[[GROUP]] : tensor<5xi32>
// CHECK:        }
func.func @test_inputs_with_multiple_uses(%arg0 : tensor<5xi32>) -> tensor<5xi32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %1 = tcp.const {value = dense<[3, 6, 10, 15, 23]> : tensor<5xi32>} : tensor<5xi32>
      %2 = tcp.add %arg0, %1 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      %3 = tcp.mul %2, %arg0 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      tcp.yield %3 : tensor<5xi32>
  }) : () -> tensor<5xi32>
   return %10 : tensor<5xi32>
}
