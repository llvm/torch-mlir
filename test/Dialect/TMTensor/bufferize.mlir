// RUN: torch-mlir-opt -split-input-file -tm-tensor-bufferize %s | FileCheck %s

// Each tensor operand crosses to a buffer at the TMTensor op that reads it, not
// at the operand's definition, so a later full bufferization can see the read.
// See bufferize-one-shot-hazard.mlir for the end-to-end consequence.

// -----
// CHECK-LABEL:   func.func @scan_1d_inclusive(
// CHECK-SAME:            %[[IN_TENSOR:.*]]: tensor<128xi32>, %[[OUT_TENSOR:.*]]: tensor<128xi32>,
// CHECK-SAME:            %[[ACC_TENSOR:.*]]: tensor<i32>) -> (tensor<128xi32>, tensor<i32>) {
// The payload reads neither `out` nor `acc`, so neither incoming value is
// copied. CHECK-NEXT from here to the op is what makes that absence
// enforceable: a bare CHECK-NOT would only constrain the gap between the two
// positive matches bracketing it, leaving the rest of the prologue free to hold
// a copy.
// CHECK-NEXT:      %[[OUT_MEMREF:.*]] = memref.alloc() : memref<128xi32>
// CHECK-NEXT:      %[[OUT_TENSOR_NEW:.*]] = bufferization.to_tensor %[[OUT_MEMREF]] restrict : memref<128xi32> to tensor<128xi32>
// CHECK-NEXT:      %[[ACC_MEMREF:.*]] = memref.alloc() : memref<i32>
// CHECK-NEXT:      %[[ACC_TENSOR_NEW:.*]] = bufferization.to_tensor %[[ACC_MEMREF]] restrict : memref<i32> to tensor<i32>
// CHECK-NEXT:      %[[IN_MEMREF:.*]] = bufferization.to_buffer %[[IN_TENSOR]] read_only : tensor<128xi32> to memref<128xi32>
// CHECK-NEXT:      tm_tensor.scan dimension(0) inclusive(true) ins(%[[IN_MEMREF]] : memref<128xi32>)
// CHECK-SAME:            outs(%[[OUT_MEMREF]], %[[ACC_MEMREF]] : memref<128xi32>, memref<i32>) {
// CHECK:           ^bb0(%[[OUT_PREV_ELEMENT:.*]]: i32, %[[IN_ELEMENT:.*]]: i32):
// CHECK:             %[[OUT_CURRENT_ELEMENT:.*]] = arith.addi %[[OUT_PREV_ELEMENT]], %[[IN_ELEMENT]] : i32
// CHECK:             tm_tensor.yield %[[OUT_CURRENT_ELEMENT]] : i32
// CHECK:           }
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
// CHECK:           %[[OUT_MEMREF:.*]] = memref.alloc() : memref<128xi32>
// CHECK:           %[[OUT_TENSOR_NEW:.*]] = bufferization.to_tensor %[[OUT_MEMREF]] restrict : memref<128xi32> to tensor<128xi32>
// CHECK:           %[[ACC_MEMREF:.*]] = memref.alloc() : memref<i32>
// CHECK:           %[[ACC_TENSOR_NEW:.*]] = bufferization.to_tensor %[[ACC_MEMREF]] restrict : memref<i32> to tensor<i32>
// The payload reads `acc`, so its incoming value is copied into the buffer the
// op updates in place -- straight from the tensor operand, so exactly once.
// CHECK:           bufferization.materialize_in_destination %[[ACC_TENSOR]] in restrict writable %[[ACC_MEMREF]] : (tensor<i32>, memref<i32>) -> ()
// CHECK:           %[[IN_MEMREF:.*]] = bufferization.to_buffer %[[IN_TENSOR]] read_only : tensor<128xi32> to memref<128xi32>
// CHECK:           tm_tensor.scan dimension(0) inclusive(false) ins(%[[IN_MEMREF]] : memref<128xi32>)
// CHECK-SAME:            outs(%[[OUT_MEMREF]], %[[ACC_MEMREF]] : memref<128xi32>, memref<i32>) {
// CHECK:           ^bb0(%[[OUT_PREV_ELEMENT:.*]]: i32, %[[IN_ELEMENT:.*]]: i32):
// CHECK:             %[[OUT_CURRENT_ELEMENT:.*]] = arith.addi %[[OUT_PREV_ELEMENT]], %[[IN_ELEMENT]] : i32
// CHECK:             tm_tensor.yield %[[OUT_CURRENT_ELEMENT]] : i32
// CHECK:           }
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
// CHECK-LABEL:   func.func @scan_1d_dynamic(
// CHECK-SAME:            %[[IN_TENSOR:.*]]: tensor<?xi32>, %[[OUT_TENSOR:.*]]: tensor<?xi32>,
// CHECK-SAME:            %[[ACC_TENSOR:.*]]: tensor<i32>) -> (tensor<?xi32>, tensor<i32>) {
// The size comes off the tensor argument, NOT off a buffer, and neither `outs`
// operand is read by the payload, so nothing is copied anywhere. CHECK-NEXT
// from here to the op is what makes those absences enforceable -- it leaves no
// gap for a `memref.dim` on a buffer, or for a copy, to hide in.
// CHECK-NEXT:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[OUT_DIM:.*]] = tensor.dim %[[OUT_TENSOR]], %[[C0]] : tensor<?xi32>
// CHECK-NEXT:      %[[OUT_MEMREF:.*]] = memref.alloc(%[[OUT_DIM]]) : memref<?xi32>
// CHECK-NEXT:      %[[OUT_TENSOR_NEW:.*]] = bufferization.to_tensor %[[OUT_MEMREF]] restrict : memref<?xi32> to tensor<?xi32>
// CHECK-NEXT:      %[[ACC_MEMREF:.*]] = memref.alloc() : memref<i32>
// CHECK-NEXT:      %[[ACC_TENSOR_NEW:.*]] = bufferization.to_tensor %[[ACC_MEMREF]] restrict : memref<i32> to tensor<i32>
// CHECK-NEXT:      %[[IN_MEMREF:.*]] = bufferization.to_buffer %[[IN_TENSOR]] read_only : tensor<?xi32> to memref<?xi32>
// CHECK-NEXT:      tm_tensor.scan dimension(0) inclusive(true) ins(%[[IN_MEMREF]] : memref<?xi32>)
// CHECK-SAME:            outs(%[[OUT_MEMREF]], %[[ACC_MEMREF]] : memref<?xi32>, memref<i32>) {
// CHECK:           return %[[OUT_TENSOR_NEW]], %[[ACC_TENSOR_NEW]] : tensor<?xi32>, tensor<i32>
func.func @scan_1d_dynamic(%in: tensor<?xi32>, %out: tensor<?xi32>, %acc: tensor<i32>) -> (tensor<?xi32>, tensor<i32>) {
  %ret_out, %ret_acc = tm_tensor.scan dimension(0) inclusive(true)
    ins(%in : tensor<?xi32>) outs(%out, %acc: tensor<?xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      tm_tensor.yield %sum : i32
  } -> tensor<?xi32>, tensor<i32>
  return %ret_out, %ret_acc: tensor<?xi32>, tensor<i32>
}

// -----
// CHECK-LABEL:   func.func @scatter_update_scalar_1D(
// CHECK-SAME:            %[[ORIG_TENSOR:.*]]: tensor<8xi32>,
// CHECK-SAME:            %[[INDICES_TENSOR:.*]]: tensor<3x1xi32>,
// CHECK-SAME:            %[[UPDATES_TENSOR:.*]]: tensor<3xi32>) -> tensor<8xi32> {
// CHECK:           %[[ORIG_MEMREF:.*]] = memref.alloc() : memref<8xi32>
// CHECK:           %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[ORIG_MEMREF]] restrict : memref<8xi32> to tensor<8xi32>
// CHECK:           bufferization.materialize_in_destination %[[ORIG_TENSOR]] in restrict writable %[[ORIG_MEMREF]] : (tensor<8xi32>, memref<8xi32>) -> ()
// CHECK:           %[[UPDATES_MEMREF:.*]] = bufferization.to_buffer %[[UPDATES_TENSOR]] read_only : tensor<3xi32> to memref<3xi32>
// CHECK:           %[[INDICES_MEMREF:.*]] = bufferization.to_buffer %[[INDICES_TENSOR]] read_only : tensor<3x1xi32> to memref<3x1xi32>
// CHECK:           tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true) ins(%[[UPDATES_MEMREF]], %[[INDICES_MEMREF]]
// CHECK-SAME:        : memref<3xi32>, memref<3x1xi32>) outs(%[[ORIG_MEMREF]] : memref<8xi32>) {
// CHECK:           ^bb0(%[[UPDATE_SCALAR:.*]]: i32, %[[ORIG_SCALAR:.*]]: i32):
// CHECK:             tm_tensor.yield %[[UPDATE_SCALAR]] : i32
// CHECK:           }
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
// CHECK:           %[[ORIG_MEMREF:.*]] = memref.alloc() : memref<8xi32>
// CHECK:           %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[ORIG_MEMREF]] restrict : memref<8xi32> to tensor<8xi32>
// CHECK:           bufferization.materialize_in_destination %[[ORIG_TENSOR]] in restrict writable %[[ORIG_MEMREF]] : (tensor<8xi32>, memref<8xi32>) -> ()
// CHECK:           %[[UPDATES_MEMREF:.*]] = bufferization.to_buffer %[[UPDATES_TENSOR]] read_only : tensor<3xi32> to memref<3xi32>
// CHECK:           %[[INDICES_MEMREF:.*]] = bufferization.to_buffer %[[INDICES_TENSOR]] read_only : tensor<3x1xi32> to memref<3x1xi32>
// CHECK:           tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true) ins(%[[UPDATES_MEMREF]], %[[INDICES_MEMREF]]
// CHECK-SAME:        : memref<3xi32>, memref<3x1xi32>) outs(%[[ORIG_MEMREF]] : memref<8xi32>) {
// CHECK:           ^bb0(%[[UPDATE_SCALAR:.*]]: i32, %[[ORIG_SCALAR:.*]]: i32):
// CHECK:             %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:             %[[ADD:.*]] = arith.addi %[[ORIG_SCALAR]], %[[CST1]] : i32
// CHECK:             tm_tensor.yield %[[ADD]] : i32
// CHECK:           }
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
