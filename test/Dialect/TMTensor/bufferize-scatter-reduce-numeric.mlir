// REQUIRES: mlir-runner
// RUN: torch-mlir-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func(convert-torch-to-tmtensor,convert-torch-to-linalg,canonicalize,cse,torch-finalizing-backend-type-conversion,tm-tensor-bufferize,tm-tensor-to-loops))' \
// RUN:   | mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:       --convert-linalg-to-loops --expand-strided-metadata --lower-affine \
// RUN:       --convert-scf-to-cf --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:       --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN:   | mlir-runner -e main --entry-point-result=void \
// RUN:       --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN:       --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   | FileCheck %s

// Numeric regression test for `torch.aten.scatter_reduce.two` with
// `include_self=false`, starting from the actual torch op and *executing* the
// result. This is the only level at which the bug this file guards against is
// observable as a value rather than as a shape of IR.
//
// The pipeline deliberately finishes bufferization with plain
// `--one-shot-bufferize`, so that One-Shot's analysis actually runs. Adding
// `copy-before-write` -- as `refbackend`'s pipeline does -- would skip the
// analysis and copy defensively, which hides the bug.
//
// An e2e test cannot stand in for this file. `ScatterReduceFloatSumModule`
// already *is* this op with `include_self=false`, and it passes on a build
// without the fix, for two independent reasons. First, `refbackend`'s pipeline
// passes `copy-before-write`, which skips One-Shot's analysis altogether.
// Second, it would pass even with the analysis enabled, because the pipeline
// opens with `linalg-fuse-elementwise-ops`, whose `RemoveOutsDependency`
// pattern rewrites every `outs` operand the payload does not read to a fresh
// `tensor.empty` -- dissolving the shared init described below before
// bufferization is ever consulted.
//
// Neither defence is one this pass can rely on. `copy-before-write` is opt-in,
// and the `RemoveOutsDependency` rewrite is undone by any `cse` between fusion
// and bufferization: the fresh `tensor.empty` is structurally identical to the
// one the `linalg.fill` below already writes into, so CSE merges the two and
// the shared init comes straight back. What keeps the e2e suite green is
// therefore an accident of pass ordering, not anything the suite can be
// configured to expose -- which is why this file pins the behaviour at the
// level of values instead.
//
// The hazard `tm-tensor-bufferize` is handed: `convert-torch-to-tmtensor`
// computes the reduction identity (a `linalg.fill`) and the update values (a
// `linalg.generic`) into the *same* `tensor.empty` init, then runs two
// `tm_tensor.scatter`s -- a "clear" pass that plants the identity at each target
// and a "reduce" pass that combines the updates in. One init, several writers.
//
// `cse` in the pipeline above is load-bearing, not tidying: the lowering emits
// two structurally identical `tensor.empty` + `linalg.fill` pairs, and it is CSE
// that merges them into the single init the two scatters then share. Drop it and
// this test passes even on a build without the fix.
//
// If the scatters' reads are stated at their operands' definitions rather than
// at the scatters, One-Shot Bufferize sees no conflict between those writers and
// folds them onto one buffer, so one of the two values is lost. See
// bufferize-one-shot-hazard.mlir for the IR-level property.
//
// Both modes below use the same inputs:
//
//   self  = [10 20; 30 40; 50 60; 70 80]
//   index = [[0 1]; [0 3]]      src = [[1 2]; [3 4]]
//   scatter_reduce(dim=0, include_self=false)
//
// Cell (0,0) receives both 1 and 3; (1,1) receives 2; (3,1) receives 4; every
// other cell keeps its `self` value.
//
// Two reduce modes are checked because the collapse destroys a different value
// in each, and a fix that recovered only one shape would still be wrong:
//
//   sum  -- the identity is 0.0, which is also the constant the `linalg.generic`
//           init is filled with, so CSE merges the two fills into one that
//           *precedes* the generic. The surviving value is the updates, which
//           the clear pass then scatters instead of zeros and the add pass adds
//           a second time: 2x at every singly-targeted cell. (At (0,0) it is
//           last-write plus full sum, 3 + (1+3) = 7, not 8.)
//
//   amax -- the identity is -inf, a different constant, so its fill is a third
//           writer of the shared init emitted *after* the generic. That fill is
//           what survives, destroying the updates, and both passes read -inf.
//
//   mode   correct                        collapsed
//   sum    [4 20; 30 2; 50 60; 70 4]      [7 20; 30 4; 50 60; 70 8]
//   amax   [3 20; 30 2; 50 60; 70 4]      [-inf 20; 30 -inf; 50 60; 70 -inf]
//
// CHECK:      Unranked Memref
// CHECK-NEXT: [4, 20]
// CHECK-NEXT: [30, 2]
// CHECK-NEXT: [50, 60]
// CHECK-NEXT: [70, 4]
// CHECK:      Unranked Memref
// CHECK-NEXT: [3, 20]
// CHECK-NEXT: [30, 2]
// CHECK-NEXT: [50, 60]
// CHECK-NEXT: [70, 4]

func.func private @printMemrefF32(tensor<*xf32>)

// The `torch_c` casts wrap the torch op inside a builtin-tensor signature so
// that @main can call it directly and the whole test stays in one file. They
// are erased by `torch-finalizing-backend-type-conversion`.
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

func.func @main() {
  %self = arith.constant dense<[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]> : tensor<4x2xf32>
  %index = arith.constant dense<[[0, 1], [0, 3]]> : tensor<2x2xi64>
  %src = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %sum = call @scatter_reduce_sum(%self, %index, %src)
      : (tensor<4x2xf32>, tensor<2x2xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>
  %sum_unranked = tensor.cast %sum : tensor<4x2xf32> to tensor<*xf32>
  call @printMemrefF32(%sum_unranked) : (tensor<*xf32>) -> ()

  %amax = call @scatter_reduce_amax(%self, %index, %src)
      : (tensor<4x2xf32>, tensor<2x2xi64>, tensor<2x2xf32>) -> tensor<4x2xf32>
  %amax_unranked = tensor.cast %amax : tensor<4x2xf32> to tensor<*xf32>
  call @printMemrefF32(%amax_unranked) : (tensor<*xf32>) -> ()
  return
}
