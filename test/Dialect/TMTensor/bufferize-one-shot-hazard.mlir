// RUN: torch-mlir-opt %s -tm-tensor-bufferize -tm-tensor-to-loops \
// RUN:   | mlir-opt --one-shot-bufferize="allow-unknown-ops" \
// RUN:   | FileCheck %s

// The IR shape `convert-torch-to-tmtensor` emits for
// `torch.aten.scatter_reduce.two` with `include_self=false`: one `tensor.empty`
// init with two writers -- the reduction identity and the update values -- read
// by two separate `tm_tensor.scatter`s, a "clear" pass and an "add" pass.
//
// The property pinned here: the two loops must read from two *different*
// buffers, and the buffer the clear loop reads must be the one filled with 0.0.
// See bufferize-scatter-reduce-numeric.mlir for the same bug at the level of
// values.

// CHECK-LABEL:   func.func @scatter_reduce_sum_no_include_self(
// CHECK-SAME:                            %[[INDICES:.*]]: tensor<2x1xi32>,
// CHECK-SAME:                            %[[SRC:.*]]: tensor<2xf32>,
// CHECK-SAME:                            %[[SELF:.*]]: tensor<4xf32>
// CHECK-DAG:       %[[SELF_BUF:.*]] = bufferization.to_buffer %[[SELF]]
// CHECK-DAG:       %[[SRC_BUF:.*]] = bufferization.to_buffer %[[SRC]]
// CHECK-DAG:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// The two writers of the shared init get separate buffers: neither can be
// written in place, because the scatter that reads the other one conflicts.
// CHECK:           %[[VALS:.*]] = memref.alloc(){{.*}} : memref<2xf32>
// CHECK:           %[[ZEROS:.*]] = memref.alloc(){{.*}} : memref<2xf32>
// CHECK:           linalg.fill ins(%[[ZERO]] : f32) outs(%[[ZEROS]] : memref<2xf32>)
// CHECK:           linalg.generic {{.*}} ins(%[[SRC_BUF]]{{.*}}) outs(%[[VALS]] : memref<2xf32>)

// The clear loop reads the zeros and overwrites its targets.
// CHECK:           %[[CLEARED:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           memref.copy %[[SELF_BUF]], %[[CLEARED]]
// CHECK:           %[[IDX_BUF:.*]] = bufferization.to_buffer %[[INDICES]] read_only
// CHECK:           scf.for
// CHECK:             %[[U0:.*]] = memref.load %[[ZEROS]]
// CHECK:             memref.load %[[IDX_BUF]]
// CHECK:             memref.store %[[U0]], %[[CLEARED]]
// CHECK:           }

// The add loop reads the values -- a *different* buffer.
// CHECK:           %[[RESULT:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[RESULT_TENSOR:.*]] = bufferization.to_tensor %[[RESULT]] restrict
// CHECK:           memref.copy %[[CLEARED]], %[[RESULT]]
// CHECK:           %[[IDX_BUF_2:.*]] = bufferization.to_buffer %[[INDICES]] read_only
// CHECK:           scf.for
// CHECK:             %[[U1:.*]] = memref.load %[[VALS]]
// CHECK:             memref.load %[[IDX_BUF_2]]
// CHECK:             %[[O:.*]] = memref.load %[[RESULT]]
// CHECK:             %[[SUM:.*]] = arith.addf %[[U1]], %[[O]] : f32
// CHECK:             memref.store %[[SUM]], %[[RESULT]]
// CHECK:           }
// CHECK:           return %[[RESULT_TENSOR]] : tensor<4xf32>
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
  %cleared = tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(true)
      ins(%zeros, %indices : tensor<2xf32>, tensor<2x1xi32>)
      outs(%self : tensor<4xf32>) {
  ^bb0(%update: f32, %orig: f32):
    tm_tensor.yield %update : f32
  } -> tensor<4xf32>

  // Add pass: accumulate the values on top.
  %result = tm_tensor.scatter {dimension_map = array<i64: 0>} unique_indices(false)
      ins(%vals, %indices : tensor<2xf32>, tensor<2x1xi32>)
      outs(%cleared : tensor<4xf32>) {
  ^bb0(%update: f32, %orig: f32):
    %sum = arith.addf %update, %orig : f32
    tm_tensor.yield %sum : f32
  } -> tensor<4xf32>

  return %result : tensor<4xf32>
}
