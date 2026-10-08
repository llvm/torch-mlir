// RUN: torch-mlir-opt %s -tm-tensor-bufferize | FileCheck %s
// RUN: torch-mlir-opt %s -tm-tensor-bufferize -tm-tensor-to-loops -cse \
// RUN:   -canonicalize | FileCheck %s --check-prefix=CSE

// The IR shape `convert-torch-to-tmtensor` emits for
// `torch.aten.scatter_reduce.two` with `include_self=false`: one `tensor.empty`
// init with two writers -- the reduction identity and the update values -- read
// by two separate `tm_tensor.scatter`s, a "clear" pass and an "add" pass.
//
// Each scatter's read of an `ins` operand must be stated at that scatter, after
// both writers of the shared init. Stated at the operand's definition, it
// precedes the other writer, and a later One-Shot Bufferize sees no
// read-after-write conflict and puts both writers in one buffer.

// CHECK-LABEL:   func.func @scatter_reduce_sum_no_include_self(
// CHECK:           %[[ZEROS:.*]] = linalg.fill
// CHECK-NOT:       bufferization.to_buffer %[[ZEROS]]
// CHECK:           %[[VALS:.*]] = linalg.generic
// CHECK:           %[[ZEROS_BUF:.*]] = bufferization.to_buffer %[[ZEROS]] read_only
// CHECK:           tm_tensor.scatter {{.*}} unique_indices(true) ins(%[[ZEROS_BUF]],
// CHECK:           %[[VALS_BUF:.*]] = bufferization.to_buffer %[[VALS]] read_only
// CHECK:           tm_tensor.scatter {{.*}} unique_indices(false) ins(%[[VALS_BUF]],
func.func @scatter_reduce_sum_no_include_self(
    %indices: tensor<2x1xi32>, %src: tensor<2xf32>,
    %self: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2xf32>

  // Writer 1: the reduction identity.
  %zeros = linalg.fill ins(%zero : f32) outs(%init : tensor<2xf32>) -> tensor<2xf32>

  // Writer 2: the update values, sharing the SAME init tensor.
  %vals = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%src : tensor<2xf32>) outs(%init : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2xf32>

  // Clear pass: plant the identity at every scatter target.
  %cleared = tm_tensor.scatter <dimension_map = [0]> unique_indices(true)
      ins(%zeros, %indices : tensor<2xf32>, tensor<2x1xi32>)
      outs(%self : tensor<4xf32>) {
  ^bb0(%update: f32, %orig: f32):
    tm_tensor.yield %update : f32
  } -> tensor<4xf32>

  // Add pass: accumulate the values on top.
  %result = tm_tensor.scatter <dimension_map = [0]> unique_indices(false)
      ins(%vals, %indices : tensor<2xf32>, tensor<2x1xi32>)
      outs(%cleared : tensor<4xf32>) {
  ^bb0(%update: f32, %orig: f32):
    %sum = arith.addf %update, %orig : f32
    tm_tensor.yield %sum : f32
  } -> tensor<4xf32>

  return %result : tensor<4xf32>
}

// One value read by two TMTensor ops, with an in-place write of it in between.
// CSE merges the two identical `to_buffer %t read_only` crossings into the
// first, which moves the second op's read before the write. `canonicalize`
// then rewrites each loop's `memref.load` of a `read_only` crossing into a
// `tensor.extract` at the load, so the second loop reads %t after the fill.

// CSE-LABEL:   func.func @read_write_read(
// CSE:           %[[T:.*]] = linalg.generic
// CSE:           scf.for
// CSE:             tensor.extract %[[T]]
// CSE:           linalg.fill ins(%{{.*}} : f32) outs(%[[T]] : tensor<4xf32>)
// CSE:           scf.for
// CSE:             tensor.extract %[[T]]
// CSE:           return
func.func @read_write_read(%in: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %one = arith.constant 1.000000e+00 : f32
  %zero = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4xf32>
  %t = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%in : tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %xp = arith.addf %x, %one : f32
    linalg.yield %xp : f32
  } -> tensor<4xf32>

  // First read of %t.
  %out1 = tensor.empty() : tensor<4xf32>
  %acc1 = tensor.empty() : tensor<f32>
  %r1, %a1 = tm_tensor.scan dimension(0) inclusive(true)
      ins(%t : tensor<4xf32>) outs(%out1, %acc1 : tensor<4xf32>, tensor<f32>) {
  ^bb0(%x: f32, %y: f32):
    %s = arith.addf %x, %y : f32
    tm_tensor.yield %s : f32
  } -> tensor<4xf32>, tensor<f32>

  // In-place writer of %t's buffer, between the two reads.
  %zeros = linalg.fill ins(%zero : f32) outs(%t : tensor<4xf32>) -> tensor<4xf32>

  // Second read of %t.
  %out2 = tensor.empty() : tensor<4xf32>
  %acc2 = tensor.empty() : tensor<f32>
  %r2, %a2 = tm_tensor.scan dimension(0) inclusive(true)
      ins(%t : tensor<4xf32>) outs(%out2, %acc2 : tensor<4xf32>, tensor<f32>) {
  ^bb0(%x: f32, %y: f32):
    %s = arith.addf %x, %y : f32
    tm_tensor.yield %s : f32
  } -> tensor<4xf32>, tensor<f32>

  return %r1, %r2, %zeros : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}
