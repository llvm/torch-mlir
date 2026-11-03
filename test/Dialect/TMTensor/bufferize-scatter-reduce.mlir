// RUN: torch-mlir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(convert-torch-to-tmtensor,convert-torch-to-linalg,canonicalize,cse,torch-finalizing-backend-type-conversion,tm-tensor-bufferize))' \
// RUN:   | FileCheck %s

// `torch.aten.scatter_reduce.two` with `include_self=false`, lowered from the
// torch op. `convert-torch-to-tmtensor` computes the reduction identity and the
// update values into one `tensor.empty` init, then runs two
// `tm_tensor.scatter`s: a "clear" pass that plants the identity at each target
// and a "reduce" pass that combines the updates in. The `cse` in the pipeline is
// what merges the lowering's two identical `tensor.empty` + `linalg.fill` pairs
// into that shared init.
//
// Each scatter's read must be stated at the scatter, after every writer of the
// shared init. The two modes below have different writers:
//
//   sum  -- the identity is 0.0, the same constant the update `linalg.generic`
//           init is filled with, so the zeros fill is the generic's `outs`. The
//           generic writes over the zeros, and the clear scatter's read of them
//           must come after it.
//
//   amax -- the identity is -inf, a different constant, so its fill is a third
//           writer of the init, emitted after the generic. The reduce scatter's
//           read of the update values must come after that fill.

// CHECK-LABEL:   func.func @scatter_reduce_sum(
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[ZEROS:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[INIT]] : tensor<4xf32>)
// CHECK-NOT:       bufferization.to_buffer %[[ZEROS]]
// CHECK:           %[[VALS:.*]]:3 = linalg.generic {{.*}} outs({{.*}}, %[[ZEROS]] :
// CHECK:           %[[ZEROS_BUF:.*]] = bufferization.to_buffer %[[ZEROS]] read_only
// CHECK:           tm_tensor.scatter {{.*}} ins(%[[ZEROS_BUF]],
// CHECK:           %[[VALS_BUF:.*]] = bufferization.to_buffer %[[VALS]]#2 read_only
// CHECK:           tm_tensor.scatter {{.*}} ins(%[[VALS_BUF]],
func.func @scatter_reduce_sum(%self_b: tensor<4x2xf32>, %index_b: tensor<2x2xi64>,
                              %src_b: tensor<2x2xf32>) -> tensor<4x2xf32> {
  %self = torch_c.from_builtin_tensor %self_b : tensor<4x2xf32> -> !torch.vtensor<[4,2],f32>
  %index = torch_c.from_builtin_tensor %index_b : tensor<2x2xi64> -> !torch.vtensor<[2,2],si64>
  %src = torch_c.from_builtin_tensor %src_b : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>
  %dim = torch.constant.int 0
  %reduce = torch.constant.str "sum"
  %include_self = torch.constant.bool false
  %0 = torch.aten.scatter_reduce.two %self, %dim, %index, %src, %reduce, %include_self :
      !torch.vtensor<[4,2],f32>, !torch.int, !torch.vtensor<[2,2],si64>,
      !torch.vtensor<[2,2],f32>, !torch.str, !torch.bool -> !torch.vtensor<[4,2],f32>
  %r = torch_c.to_builtin_tensor %0 : !torch.vtensor<[4,2],f32> -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}

// CHECK-LABEL:   func.func @scatter_reduce_amax(
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VALS:.*]]:3 = linalg.generic
// CHECK-NOT:       bufferization.to_buffer %[[VALS]]#2
// CHECK:           %[[IDENT:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[INIT]] : tensor<4xf32>)
// CHECK:           %[[IDENT_BUF:.*]] = bufferization.to_buffer %[[IDENT]] read_only
// CHECK:           tm_tensor.scatter {{.*}} ins(%[[IDENT_BUF]],
// CHECK:           %[[VALS_BUF:.*]] = bufferization.to_buffer %[[VALS]]#2 read_only
// CHECK:           tm_tensor.scatter {{.*}} ins(%[[VALS_BUF]],
func.func @scatter_reduce_amax(%self_b: tensor<4x2xf32>, %index_b: tensor<2x2xi64>,
                               %src_b: tensor<2x2xf32>) -> tensor<4x2xf32> {
  %self = torch_c.from_builtin_tensor %self_b : tensor<4x2xf32> -> !torch.vtensor<[4,2],f32>
  %index = torch_c.from_builtin_tensor %index_b : tensor<2x2xi64> -> !torch.vtensor<[2,2],si64>
  %src = torch_c.from_builtin_tensor %src_b : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>
  %dim = torch.constant.int 0
  %reduce = torch.constant.str "amax"
  %include_self = torch.constant.bool false
  %0 = torch.aten.scatter_reduce.two %self, %dim, %index, %src, %reduce, %include_self :
      !torch.vtensor<[4,2],f32>, !torch.int, !torch.vtensor<[2,2],si64>,
      !torch.vtensor<[2,2],f32>, !torch.str, !torch.bool -> !torch.vtensor<[4,2],f32>
  %r = torch_c.to_builtin_tensor %0 : !torch.vtensor<[4,2],f32> -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}
