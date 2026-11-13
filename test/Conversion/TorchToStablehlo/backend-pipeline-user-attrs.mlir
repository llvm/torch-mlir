// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-stablehlo-backend-pipeline)' -split-input-file | FileCheck %s

// Verify that user annotations survive the full StableHLO backend pipeline,
// including the upstream StableHLO rewrites that run at its tail
// (`stablehlo-canonicalize-dynamism` folding `stablehlo.dynamic_reshape` into
// `stablehlo.reshape`, and `stablehlo-legalize-deprecated-ops` rewriting
// `stablehlo.dot` into `stablehlo.dot_general`). The per-conversion tests in
// linear.mlir and view_like.mlir stop before those rewrites. See
// docs/user_annotations.md.

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
