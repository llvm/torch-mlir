// RUN: npcomp-opt -lower-structural-to-memref <%s | FileCheck %s --dump-input=fail

// Basic cases.

// CHECK-LABEL: func @identity(%arg0: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    return %arg0 : memref<?xf32>
func @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}

// CHECK-LABEL: func @bb_arg(%arg0: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    br ^bb1(%arg0 : memref<?xf32>)
// CHECK-NEXT:  ^bb1(%[[BBARG:.*]]: memref<?xf32>):
// CHECK-NEXT:    return %[[BBARG]] : memref<?xf32>
func @bb_arg(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  br ^bb1(%arg0: tensor<?xf32>)
^bb1(%bbarg: tensor<?xf32>):
  return %bbarg : tensor<?xf32>
}

// CHECK-LABEL: func @select(%arg0: i1, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    %[[RET:.*]] = select %arg0, %arg1, %arg2 : memref<?xf32>
// CHECK-NEXT:    return %[[RET]] : memref<?xf32>
func @select(%pred: i1, %true_val: tensor<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
  %0 = std.select %pred, %true_val, %false_val : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @if(%arg0: i1, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    %[[RET:.*]] = scf.if %arg0 -> (memref<?xf32>) {
// CHECK-NEXT:      scf.yield %arg1 : memref<?xf32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2 : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RET]] : memref<?xf32>
func @if(%pred: i1, %true_val: tensor<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
  %0 = scf.if %pred -> (tensor<?xf32>) {
    scf.yield %true_val : tensor<?xf32>
  } else {
    scf.yield %false_val : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @for(%arg0: memref<f32>, %arg1: index, %arg2: index, %arg3: index) -> memref<f32> {
// CHECK-NEXT:    %[[RET:.*]] = scf.for %arg4 = %arg1 to %arg2 step %arg3 iter_args(%arg5 = %arg0) -> (memref<f32>) {
// CHECK-NEXT:      scf.yield %arg5 : memref<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RET]] : memref<f32>
// CHECK-NEXT:  }
func @for(%arg0: tensor<f32>, %lb: index, %ub: index, %step: index) -> tensor<f32> {
  %ret = scf.for %iv = %lb to %ub step %step iter_args(%iter = %arg0) -> tensor<f32> {
    scf.yield %iter : tensor<f32>
  }
  return %ret : tensor<f32>
}


// Test the interactions with materializations.
// Note: this pass never actually expects IR with memref argument types.
// We use memref-typed arguments purely for testing convenience.

// CHECK-LABEL: func @identity_materializations(%arg0: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    return %arg0 : memref<?xf32>
func @identity_materializations(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tensor_to_memref %arg0 : memref<?xf32>
  %1 = tensor_load %0 : memref<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func @if_materializations(%arg0: i1, %arg1: memref<?xf32>, %arg2: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    %[[RET:.*]] = scf.if %arg0 -> (memref<?xf32>) {
// CHECK-NEXT:      scf.yield %arg1 : memref<?xf32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg2 : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[RET]] : memref<?xf32>
func @if_materializations(%pred: i1, %true_val_memref: memref<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
  %true_val = tensor_load %true_val_memref : memref<?xf32>
  %0 = scf.if %pred -> (tensor<?xf32>) {
    scf.yield %true_val : tensor<?xf32>
  } else {
    scf.yield %false_val : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @elide_tensor_load(%arg0: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    return %arg0 : memref<?xf32>
func @elide_tensor_load(%arg0: memref<?xf32>) -> tensor<?xf32> {
  %0 = tensor_load %arg0 : memref<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @elide_tensor_to_memref(%arg0: memref<?xf32>) -> memref<?xf32> {
// CHECK-NEXT:    return %arg0 : memref<?xf32>
func @elide_tensor_to_memref(%arg0: tensor<?xf32>) -> memref<?xf32> {
  %0 = tensor_to_memref %arg0 : memref<?xf32>
  return %0 : memref<?xf32>
}
