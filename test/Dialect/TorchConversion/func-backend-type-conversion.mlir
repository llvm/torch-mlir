// RUN: torch-mlir-opt %s -torch-func-backend-type-conversion -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// This test is largely copied from `func-bufferize` upstream, as it covers
// the same scope.

// CHECK-LABEL:   func.func @identity(
// CHECK-SAME:                   %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           return %[[ARG]] : tensor<f32>
func.func @identity(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  return %arg0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func @block_arguments(
// CHECK-SAME:        %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           cf.br ^bb1(%[[ARG]] : tensor<f32>)
// CHECK:         ^bb1(%[[BBARG:.*]]: tensor<f32>):
// CHECK:           return %[[BBARG]] : tensor<f32>
func.func @block_arguments(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  cf.br ^bb1(%arg0: !torch.vtensor<[],f32>)
^bb1(%bbarg: !torch.vtensor<[],f32>):
  return %bbarg : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func.func private @source() -> tensor<f32>
// CHECK-LABEL:   func.func @call_source() -> tensor<f32> {
// CHECK:           %[[RET:.*]] = call @source() : () -> tensor<f32>
// CHECK:           return %[[RET]] : tensor<f32>
func.func private @source() -> !torch.vtensor<[],f32>
func.func @call_source() -> !torch.vtensor<[],f32> {
  %0 = call @source() : () -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
// CHECK-LABEL:   func.func @call_sink(
// CHECK-SAME:                    %[[ARG:.*]]: tensor<f32>) {
// CHECK:           call @sink(%[[ARG]]) : (tensor<f32>) -> ()
// CHECK:           return
func.func private @sink(!torch.vtensor<[],f32>)
func.func @call_sink(%arg0: !torch.vtensor<[],f32>) {
  call @sink(%arg0) : (!torch.vtensor<[],f32>) -> ()
  return
}

// CHECK-LABEL:   func.func @unconverted_op_in_body() -> tensor<f32> {
// CHECK:           %[[TENSOR:.*]] = "test.source"() : () -> !torch.vtensor<[],f32>
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           return %[[BUILTIN_TENSOR]] : tensor<f32>
func.func @unconverted_op_in_body() -> !torch.vtensor<[],f32> {
  %0 = "test.source"() : () -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// Because this pass updates block arguments, it needs to also atomically
// update all terminators and issue an error if that is not possible.
func.func @unable_to_update_terminator(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
    %0 = arith.constant true
    cf.cond_br %0, ^bb1(%arg0: !torch.vtensor<[],f32>), ^bb2(%arg0: !torch.vtensor<[],f32>)
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
func.func @bwhile(%arg0: i64, %arg1: i64) -> i64 {
  %c2_i64 = arith.constant 2 : i64
  %0:2 = scf.while (%arg2 = %arg0) : (i64) -> (i64, i64) {
    %1 = arith.cmpi slt, %arg2, %arg1 : i64
    scf.condition(%1) %arg2, %arg2 : i64, i64
  } do {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = arith.muli %arg3, %c2_i64 : i64
    scf.yield %1 : i64
  }
  return %0#1 : i64
}

// Do a basic check of other types. Under the hood they all take the same
// code paths as for !torch.vtensor, so we just spot-check them here.

// CHECK-LABEL:   func.func @identity$torch.bool(
// CHECK-SAME:                   %[[ARG:.*]]: i1) -> i1 {
// CHECK:           return %[[ARG]] : i1
func.func @identity$torch.bool(%arg0: !torch.bool) -> !torch.bool {
  return %arg0 : !torch.bool
}

// CHECK-LABEL:   func.func @identity$torch.int(
// CHECK-SAME:                             %[[ARG:.*]]: i64) -> i64 {
// CHECK:           return %[[ARG]] : i64
func.func @identity$torch.int(%arg0: !torch.int) -> !torch.int {
  return %arg0 : !torch.int
}

// CHECK-LABEL:   func.func @identity$torch.float(
// CHECK-SAME:                               %[[ARG:.*]]: f64) -> f64 {
// CHECK:           return %[[ARG]] : f64
func.func @identity$torch.float(%arg0: !torch.float) -> !torch.float {
  return %arg0 : !torch.float
}

// CHECK-LABEL:   func.func @identity$torch.Generator(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i64) -> i64 {
// CHECK:           return %[[VAL_0]] : i64
func.func @identity$torch.Generator(%arg0: !torch.Generator) -> !torch.Generator {
  return %arg0 : !torch.Generator
}
