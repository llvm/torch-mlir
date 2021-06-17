// RUN: npcomp-opt %s -torch-func-backend-type-conversion -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// This test is largely copied from `func-bufferize` upstream, as it covers
// the same scope.

// CHECK-LABEL:   func @identity(
// CHECK-SAME:                   %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[TENSOR:.*]] = torch.from_builtin_tensor %[[ARG]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[MEMREF:.*]] = torch.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           return %[[MEMREF]] : tensor<f32>
func @identity(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  return %arg0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func @block_arguments(
// CHECK-SAME:        %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[T1:.*]] = torch.from_builtin_tensor %[[ARG]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[M1:.*]] = torch.to_builtin_tensor %[[T1]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           br ^bb1(%[[M1]] : tensor<f32>)
// CHECK:         ^bb1(%[[BBARG:.*]]: tensor<f32>):
// CHECK:           %[[T2:.*]] = torch.from_builtin_tensor %[[BBARG]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[M2:.*]] = torch.to_builtin_tensor %[[T2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           return %[[M2]] : tensor<f32>
func @block_arguments(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  br ^bb1(%arg0: !torch.vtensor<[],f32>)
^bb1(%bbarg: !torch.vtensor<[],f32>):
  return %bbarg : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func private @source() -> tensor<f32>
// CHECK-LABEL:   func @call_source() -> tensor<f32> {
// CHECK:           %[[RET:.*]] = call @source() : () -> tensor<f32>
// CHECK:           return %[[RET]] : tensor<f32>
func private @source() -> !torch.vtensor<[],f32>
func @call_source() -> !torch.vtensor<[],f32> {
  %0 = call @source() : () -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
// CHECK-LABEL:   func @call_sink(
// CHECK-SAME:                    %[[ARG:.*]]: tensor<f32>) {
// CHECK:           %[[TENSOR:.*]] = torch.from_builtin_tensor %[[ARG]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[MEMREF:.*]] = torch.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           call @sink(%[[MEMREF]]) : (tensor<f32>) -> ()
// CHECK:           return
func private @sink(!torch.vtensor<[],f32>)
func @call_sink(%arg0: !torch.vtensor<[],f32>) {
  call @sink(%arg0) : (!torch.vtensor<[],f32>) -> ()
  return
}

// CHECK-LABEL:   func @unconverted_op_in_body() -> tensor<f32> {
// CHECK:           %[[TENSOR:.*]] = "test.source"() : () -> !torch.vtensor<[],f32>
// CHECK:           %[[MEMREF:.*]] = torch.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           return %[[MEMREF]] : tensor<f32>
func @unconverted_op_in_body() -> !torch.vtensor<[],f32> {
  %0 = "test.source"() : () -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// Because this pass updates block arguments, it needs to also atomically
// update all terminators and issue an error if that is not possible.
func @unable_to_update_terminator(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
    %0 = constant true
    cond_br %0, ^bb1(%arg0: !torch.vtensor<[],f32>), ^bb2(%arg0: !torch.vtensor<[],f32>)
  ^bb1(%bbarg0: !torch.vtensor<[],f32>):
    // expected-error @+1 {{failed to legalize operation 'test.terminator'}}
    "test.terminator"() : () -> ()
  ^bb2(%bbarg1: !torch.vtensor<[],f32>):
    return %bbarg1 : !torch.vtensor<[],f32>
}

// -----

// There was a bug in func-bufferize pass which caused terminators without
// ReturnLike and BranchOpInterface traits (e.g. scf.condition) to always
// fail to legalize even if bufferization doesn't needed.
// Check the pass succedeed.
// CHECK: while
// CHECK: scf.while
// CHECK: scf.condition
func @bwhile(%arg0: i64, %arg1: i64) -> i64 {
  %c2_i64 = constant 2 : i64
  %0:2 = scf.while (%arg2 = %arg0) : (i64) -> (i64, i64) {
    %1 = cmpi slt, %arg2, %arg1 : i64
    scf.condition(%1) %arg2, %arg2 : i64, i64
  } do {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = muli %arg3, %c2_i64 : i64
    scf.yield %1 : i64
  }
  return %0#1 : i64
}

// Do a basic check of other types. Under the hood they all take the same
// code paths as for !torch.vtensor, so we just spot-check them here.

// CHECK-LABEL:   func @identity$torch.bool(
// CHECK-SAME:                   %[[ARG:.*]]: i1) -> i1 {
// CHECK:           %[[TORCH_BOOL:.*]] = torch.from_i1 %[[ARG]]
// CHECK:           %[[I1:.*]] = torch.to_i1 %[[TORCH_BOOL]]
// CHECK:           return %[[I1]] : i1
func @identity$torch.bool(%arg0: !torch.bool) -> !torch.bool {
  return %arg0 : !torch.bool
}

// CHECK-LABEL:   func @identity$torch.int(
// CHECK-SAME:                             %[[ARG:.*]]: i64) -> i64 {
// CHECK:           %[[TORCH_INT:.*]] = torch.from_i64 %[[ARG]]
// CHECK:           %[[I64:.*]] = torch.to_i64 %[[TORCH_INT]]
// CHECK:           return %[[I64]] : i64
func @identity$torch.int(%arg0: !torch.int) -> !torch.int {
  return %arg0 : !torch.int
}

// CHECK-LABEL:   func @identity$torch.float(
// CHECK-SAME:                               %[[ARG:.*]]: f64) -> f64 {
// CHECK:           %[[TORCH_FLOAT:.*]] = torch.from_f64 %[[ARG]]
// CHECK:           %[[F64:.*]] = torch.to_f64 %[[TORCH_FLOAT]]
// CHECK:           return %[[F64]] : f64
func @identity$torch.float(%arg0: !torch.float) -> !torch.float {
  return %arg0 : !torch.float
}
