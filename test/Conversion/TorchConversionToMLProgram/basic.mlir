// RUN: torch-mlir-opt %s -convert-torch-conversion-to-mlprogram -split-input-file | FileCheck %s

// CHECK-LABEL:   ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
// CHECK-LABEL:   func.func @f() -> i64 {
// CHECK:           %[[GLOBAL:.*]] = ml_program.global_load @global_seed : tensor<i64>
// CHECK:           %[[SEED:.*]] = tensor.extract %[[GLOBAL]][] : tensor<i64>
// CHECK:           %[[MULTIPLIER:.*]] = arith.constant 6364136223846793005 : i64
// CHECK:           %[[INC:.*]] = arith.constant 1442695040888963407 : i64
// CHECK:           %[[MUL:.*]] = arith.muli %[[SEED]], %[[MULTIPLIER]] : i64
// CHECK:           %[[NEXT_SEED:.*]] = arith.addi %[[MUL]], %[[INC]] : i64
// CHECK:           %[[INSERTED:.*]] = tensor.insert %[[NEXT_SEED]] into %[[GLOBAL]][] : tensor<i64>
// CHECK:           ml_program.global_store @global_seed = %[[INSERTED]] : tensor<i64>
// CHECK:           return %2 : i64
module {
  func.func @f() -> i64 {
    %seed = torch_c.get_next_seed : () -> i64
    return %seed : i64
  }
}
