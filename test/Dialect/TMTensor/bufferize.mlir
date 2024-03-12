// RUN: torch-mlir-opt -split-input-file -tm-tensor-bufferize %s | FileCheck %s

// -----
// CHECK-LABEL:   func.func @scan_1d_inclusive(
// CHECK-SAME:            %[[IN_TENSOR:.*]]: tensor<128xi32>, %[[OUT_TENSOR:.*]]: tensor<128xi32>,
// CHECK-SAME:            %[[ACC_TENSOR:.*]]: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
// CHECK:           %[[IN_MEMREF:.*]] = bufferization.to_memref %[[IN_TENSOR]] : memref<128xi32>
// CHECK:           %[[OUT_MEMREF_NEW:.*]] = memref.alloc() : memref<128xi32>
// CHECK:           %[[ACC_MEMREF_NEW:.*]] = memref.alloc() : memref<i32>
// CHECK:           tm_tensor.scan dimension(0) inclusive(true) ins(%[[IN_MEMREF]] : memref<128xi32>)
// CHECK-SAME:            outs(%[[OUT_MEMREF_NEW]], %[[ACC_MEMREF_NEW]] : memref<128xi32>, memref<i32>) {
// CHECK:           ^bb0(%[[OUT_PREV_ELEMENT:.*]]: i32, %[[IN_ELEMENT:.*]]: i32):
// CHECK:             %[[OUT_CURRENT_ELEMENT:.*]] = arith.addi %[[OUT_PREV_ELEMENT]], %[[IN_ELEMENT]] : i32
// CHECK:             tm_tensor.yield %[[OUT_CURRENT_ELEMENT]] : i32
// CHECK:           }
// CHECK:           %[[OUT_TENSOR_NEW:.*]] = bufferization.to_tensor %[[OUT_MEMREF_NEW]] : memref<128xi32>
// CHECK:           %[[ACC_TENSOR_NEW:.*]] = bufferization.to_tensor %[[ACC_MEMREF_NEW]] : memref<i32>
// CHECK:           return %[[OUT_TENSOR_NEW]], %[[ACC_TENSOR_NEW]] : tensor<128xi32>, tensor<i32>
func.func @scan_1d_inclusive(%in: tensor<128xi32>, %out: tensor<128xi32>, %acc: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
  %ret_out, %ret_acc = tm_tensor.scan dimension(0) inclusive(true)
    ins(%in : tensor<128xi32>) outs(%out, %acc: tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      tm_tensor.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %ret_out, %ret_acc: tensor<128xi32>, tensor<i32>
}

// -----
// CHECK-LABEL:   func.func @scan_1d_exclusive(
// CHECK-SAME:            %[[IN_TENSOR:.*]]: tensor<128xi32>, %[[OUT_TENSOR:.*]]: tensor<128xi32>,
// CHECK-SAME:            %[[ACC_TENSOR:.*]]: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
// CHECK:           %[[IN_MEMREF:.*]] = bufferization.to_memref %[[IN_TENSOR]] : memref<128xi32>
// CHECK:           %[[ACC_MEMREF:.*]] = bufferization.to_memref %[[ACC_TENSOR]] : memref<i32>
// CHECK:           %[[OUT_MEMREF_NEW:.*]] = memref.alloc() : memref<128xi32>
// CHECK:           %[[ACC_MEMREF_NEW:.*]] = memref.alloc() : memref<i32>
// CHECK:           memref.copy %[[ACC_MEMREF]], %[[ACC_MEMREF_NEW]] : memref<i32> to memref<i32>
// CHECK:           tm_tensor.scan dimension(0) inclusive(false) ins(%[[IN_MEMREF]] : memref<128xi32>)
// CHECK-SAME:            outs(%[[OUT_MEMREF_NEW]], %[[ACC_MEMREF_NEW]] : memref<128xi32>, memref<i32>) {
// CHECK:           ^bb0(%[[OUT_PREV_ELEMENT:.*]]: i32, %[[IN_ELEMENT:.*]]: i32):
// CHECK:             %[[OUT_CURRENT_ELEMENT:.*]] = arith.addi %[[OUT_PREV_ELEMENT]], %[[IN_ELEMENT]] : i32
// CHECK:             tm_tensor.yield %[[OUT_CURRENT_ELEMENT]] : i32
// CHECK:           }
// CHECK:           %[[OUT_TENSOR_NEW:.*]] = bufferization.to_tensor %[[OUT_MEMREF_NEW]] : memref<128xi32>
// CHECK:           %[[ACC_TENSOR_NEW:.*]] = bufferization.to_tensor %[[ACC_MEMREF_NEW]] : memref<i32>
// CHECK:           return %[[OUT_TENSOR_NEW]], %[[ACC_TENSOR_NEW]] : tensor<128xi32>, tensor<i32>
func.func @scan_1d_exclusive(%in: tensor<128xi32>, %out: tensor<128xi32>, %acc: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
  %ret_out, %ret_acc = tm_tensor.scan dimension(0) inclusive(false)
    ins(%in : tensor<128xi32>) outs(%out, %acc: tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      tm_tensor.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %ret_out, %ret_acc: tensor<128xi32>, tensor<i32>
}

// -----
// CHECK-LABEL:   func.func @scatter_update_scalar_1D(
// CHECK-SAME:            %[[ORIG_TENSOR:.*]]: tensor<8xi32>,
// CHECK-SAME:            %[[INDICES_TENSOR:.*]]: tensor<3x1xi32>,
// CHECK-SAME:            %[[UPDATES_TENSOR:.*]]: tensor<3xi32>) -> tensor<8xi32> {
// CHECK:           %[[UPDATES_MEMREF:.*]] = bufferization.to_memref %[[UPDATES_TENSOR]] : memref<3xi32>
// CHECK:           %[[INDICES_MEMREF:.*]] = bufferization.to_memref %[[INDICES_TENSOR]] : memref<3x1xi32>
// CHECK:           %[[ORIG_MEMREF:.*]] = bufferization.to_memref %[[ORIG_TENSOR]] : memref<8xi32>
// CHECK:           %[[ORIG_MEMREF_NEW:.*]] = memref.alloc() : memref<8xi32>
// CHECK:           memref.copy %[[ORIG_MEMREF]], %[[ORIG_MEMREF_NEW]] : memref<8xi32> to memref<8xi32>
// CHECK:           tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true) ins(%[[UPDATES_MEMREF]], %[[INDICES_MEMREF]]
// CHECK-SAME:        : memref<3xi32>, memref<3x1xi32>) outs(%[[ORIG_MEMREF_NEW]] : memref<8xi32>) {
// CHECK:           ^bb0(%[[UPDATE_SCALAR:.*]]: i32, %[[ORIG_SCALAR:.*]]: i32):
// CHECK:             tm_tensor.yield %[[UPDATE_SCALAR]] : i32
// CHECK:           }
// CHECK:           %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[ORIG_MEMREF_NEW]] : memref<8xi32>
// CHECK:           return %[[OUT_TENSOR]] : tensor<8xi32>
func.func @scatter_update_scalar_1D(
    %original: tensor<8xi32>, %indices: tensor<3x1xi32>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
    outs(%original : tensor<8xi32>)  {
  ^bb0(%update: i32, %orig: i32):  // no predecessors
    tm_tensor.yield %update: i32
  } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL:   func.func @scatter_add_scalar_1D(
// CHECK-SAME:            %[[ORIG_TENSOR:.*]]: tensor<8xi32>,
// CHECK-SAME:            %[[INDICES_TENSOR:.*]]: tensor<3x1xi32>,
// CHECK-SAME:            %[[UPDATES_TENSOR:.*]]: tensor<3xi32>) -> tensor<8xi32> {
// CHECK:           %[[UPDATES_MEMREF:.*]] = bufferization.to_memref %[[UPDATES_TENSOR]] : memref<3xi32>
// CHECK:           %[[INDICES_MEMREF:.*]] = bufferization.to_memref %[[INDICES_TENSOR]] : memref<3x1xi32>
// CHECK:           %[[ORIG_MEMREF:.*]] = bufferization.to_memref %[[ORIG_TENSOR]] : memref<8xi32>
// CHECK:           %[[ORIG_MEMREF_NEW:.*]] = memref.alloc() : memref<8xi32>
// CHECK:           memref.copy %[[ORIG_MEMREF]], %[[ORIG_MEMREF_NEW]] : memref<8xi32> to memref<8xi32>
// CHECK:           tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true) ins(%[[UPDATES_MEMREF]], %[[INDICES_MEMREF]]
// CHECK-SAME:        : memref<3xi32>, memref<3x1xi32>) outs(%[[ORIG_MEMREF_NEW]] : memref<8xi32>) {
// CHECK:           ^bb0(%[[UPDATE_SCALAR:.*]]: i32, %[[ORIG_SCALAR:.*]]: i32):
// CHECK:             %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:             %[[ADD:.*]] = arith.addi %[[ORIG_SCALAR]], %[[CST1]] : i32
// CHECK:             tm_tensor.yield %[[ADD]] : i32
// CHECK:           }
// CHECK:           %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[ORIG_MEMREF_NEW]] : memref<8xi32>
// CHECK:           return %[[OUT_TENSOR]] : tensor<8xi32>
func.func @scatter_add_scalar_1D(
    %original: tensor<8xi32>, %indices: tensor<3x1xi32>,
    %updates: tensor<3xi32>) -> tensor<8xi32> {
  %0 = tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true)
    ins(%updates, %indices : tensor<3xi32>, tensor<3x1xi32>)
    outs(%original : tensor<8xi32>)  {
  ^bb0(%update: i32, %orig: i32):  // no predecessors
    %cst1 = arith.constant 1: i32
    %add = arith.addi %orig, %cst1: i32
    tm_tensor.yield %add: i32
  } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
