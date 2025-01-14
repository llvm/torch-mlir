// RUN: torch-mlir-opt %s -convert-torch-conversion-to-mlprogram -split-input-file | FileCheck %s

module {
  func.func private @f0() -> i64
  func.func private @f1() -> i64
  func.func private @f2() -> i64
  func.func private @f3() -> i64
  func.func private @f4() -> i64
  func.func private @f5() -> i64
  func.func private @f6() -> i64
  func.func private @f7() -> i64
}

// CHECK-NOT:     ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
// CHECK-NOT: @global_seed
