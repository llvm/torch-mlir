// RUN: torch-mlir-dialects-opt <%s -convert-stablehlo-to-tcp -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tanh(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[TANH:.*]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32>
// CHECK:         return %[[TANH]] : tensor<?x?xf32>
// CHECK:       }
func.func @tanh(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.tanh %arg0 : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
