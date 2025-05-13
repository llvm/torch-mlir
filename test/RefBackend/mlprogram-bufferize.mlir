// RUN: torch-mlir-opt %s -refback-mlprogram-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   memref.global "private" @global_seed : memref<i64> = dense<0>
// CHECK-LABEL:   func.func @forward() -> i64 {
// CHECK:           %[[CST127:.*]] = arith.constant 127 : i64
// CHECK:           %[[GLOBAL_SEED:.*]] = memref.get_global @global_seed : memref<i64>
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[GLOBAL_SEED]] : memref<i64> to tensor<i64>
// CHECK:           %[[SEED:.*]] = tensor.extract %[[TENSOR]][] : tensor<i64>
// CHECK:           %[[NEXT_SEED:.*]] = arith.muli %[[SEED]], %[[CST127]] : i64
// CHECK:           %[[INSERTED:.*]] = tensor.insert %[[NEXT_SEED]] into %[[TENSOR]][] : tensor<i64>
// CHECK:           %[[GLOBAL_SEED_1:.*]] = memref.get_global @global_seed : memref<i64>
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[INSERTED]] : tensor<i64> to memref<i64>
// CHECK:           memref.copy %[[MEMREF]], %[[GLOBAL_SEED_1]] : memref<i64> to memref<i64>
// CHECK:           return %[[NEXT_SEED]] : i64
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward() -> i64 {
    %c127_i64 = arith.constant 127 : i64
    %0 = ml_program.global_load @global_seed : tensor<i64>
    %extracted = tensor.extract %0[] : tensor<i64>
    %1 = arith.muli %extracted, %c127_i64 : i64
    %inserted = tensor.insert %1 into %0[] : tensor<i64>
    ml_program.global_store @global_seed = %inserted : tensor<i64>
    return %1 : i64
  }
}

// -----

module {
  // expected-error @below {{unsupported global op type}}
  ml_program.global private mutable @global_seed(0 : i64) : i64
  func.func @forward() -> i64 {
    %c127_i64 = arith.constant 127 : i64
    %0 = ml_program.global_load @global_seed : i64
    %1 = arith.muli %0, %c127_i64 : i64
    ml_program.global_store @global_seed = %1 : i64
    return %1 : i64
  }
}

// -----

module {
  // expected-error @below {{unsupported global op type}}
  ml_program.global private mutable @global_seed(dense<0> : memref<i64>) : memref<i64>
  func.func @forward() -> i64 {
    %c127_i64 = arith.constant 127 : i64
    %0 = ml_program.global_load @global_seed : memref<i64>
    %extracted = memref.load %0[] : memref<i64>
    %1 = arith.muli %extracted, %c127_i64 : i64
    memref.store %1, %0[] : memref<i64>
    ml_program.global_store @global_seed = %0 : memref<i64>
    return %1 : i64
  }
}

// -----

module {
  // expected-error @below {{invalid tensor element type}}
  ml_program.global private mutable @global_seed(dense<0> : tensor<memref<i64>>) : tensor<memref<i64>>
  func.func @forward() -> i64 {
    %c127_i64 = arith.constant 127 : i64
    return %c127_i64 : i64
  }
}

// -----
module {
  // expected-error @below {{unimplemented: global op bufferization with dynamic shape}}
  ml_program.global private mutable @global_seed(dense<0> : tensor<1xi64>) : tensor<?xi64>
  func.func @forward() -> i64 {
    %c127_i64 = arith.constant 127 : i64
    %c0 = arith.constant 0 : index
    %0 = ml_program.global_load @global_seed : tensor<?xi64>
    %extracted = tensor.extract %0[%c0] : tensor<?xi64>
    %1 = arith.muli %extracted, %c127_i64 : i64
    %inserted = tensor.insert %1 into %0[%c0] : tensor<?xi64>
    ml_program.global_store @global_seed = %inserted : tensor<?xi64>
    return %1 : i64
  }
}
