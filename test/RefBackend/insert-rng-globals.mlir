// RUN: torch-mlir-opt %s -refback-insert-rng-globals -split-input-file | FileCheck %s

// CHECK-LABEL:   memref.global "private" @global_seed : memref<i64> = dense<0>
// CHECK-LABEL:   func @f() -> i64 {
// CHECK:           %[[MEMREF:.*]] = memref.get_global @global_seed : memref<i64>
// CHECK:           %[[SEED:.*]] = memref.load %[[MEMREF]][] : memref<i64>
// CHECK:           %[[MULTIPLIER:.*]] = arith.constant 6364136223846793005 : i64
// CHECK:           %[[INC:.*]] = arith.constant 1442695040888963407 : i64
// CHECK:           %[[MUL:.*]] = arith.muli %[[SEED]], %[[MULTIPLIER]] : i64
// CHECK:           %[[TEMP:.*]] = arith.addi %[[MUL]], %[[INC]] : i64
// CHECK:           %[[CST127:.*]] = arith.constant 127 : i64
// CHECK:           %[[NEXT_SEED:.*]] = arith.andi %[[TEMP]], %[[CST127]] : i64
// CHECK:           memref.store %[[NEXT_SEED]], %[[MEMREF]][] : memref<i64>
// CHECK:           return %[[NEXT_SEED]] : i64
module {
  func @f() -> i64 {
    %seed = torch_c.get_next_seed : () -> i64
    return %seed : i64
  }
}
