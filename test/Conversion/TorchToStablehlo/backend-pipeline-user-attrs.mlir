// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-stablehlo-backend-pipeline)' -split-input-file | FileCheck %s

// XFAIL: *
//
// Known bug: user annotations are dropped by the StableHLO rewrites that run
// at the tail of `createTorchBackendToStablehloBackendPipeline` (see
// lib/Conversion/Passes.cpp). Those passes rebuild ops via
// `rewriter.replaceOpWithNewOp<...>(...)`, which does not carry discardable
// attributes over to the new op.
//
// Two distinct passes are implicated:
//   * `stablehlo-canonicalize-dynamism` rewrites `stablehlo.dynamic_reshape`
//     into `stablehlo.reshape` (CanonicalizeDynamicReshapeOpPattern).
//   * `stablehlo-legalize-deprecated-ops` rewrites `stablehlo.dot` into
//     `stablehlo.dot_general`.
//
// Note that the per-conversion tests in linear.mlir and view_like.mlir pass,
// because they only run `-convert-torch-to-stablehlo` and stop before the
// rewrites that drop the attribute. This file exercises the whole pipeline so
// the loss is visible.
//
// This is an all-or-nothing signal: fixing only one of the two cases below
// still leaves the file failing. Remove the `XFAIL` once both hold.

// A static-shaped `aten.unsqueeze` lowers to `stablehlo.dynamic_reshape`, which
// `stablehlo-canonicalize-dynamism` then folds to `stablehlo.reshape`.
// CHECK-LABEL:  func.func @unsqueeze_user_attrs_survive_pipeline(
// CHECK:          stablehlo.reshape
// CHECK-SAME:       {mlir.user = [{tag = "unsqueeze_tag"}]}
func.func @unsqueeze_user_attrs_survive_pipeline(%arg0: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[4,3,1],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.unsqueeze %arg0, %int2 {mlir.user = [{tag = "unsqueeze_tag"}]} : !torch.vtensor<[4,3],f32>, !torch.int -> !torch.vtensor<[4,3,1],f32>
  return %0 : !torch.vtensor<[4,3,1],f32>
}

// -----

// `aten.mm` lowers to the deprecated `stablehlo.dot`, which
// `stablehlo-legalize-deprecated-ops` then rewrites to `stablehlo.dot_general`.
// CHECK-LABEL:  func.func @mm_user_attrs_survive_pipeline(
// CHECK:          stablehlo.dot_general
// CHECK-SAME:       {mlir.user = [{tag = "mm_tag"}]}
func.func @mm_user_attrs_survive_pipeline(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mm %arg0, %arg1 {mlir.user = [{tag = "mm_tag"}]} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}
