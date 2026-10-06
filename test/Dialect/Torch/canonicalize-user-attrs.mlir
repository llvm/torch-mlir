// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torchdynamo-export-to-torch-backend-pipeline)' -split-input-file | FileCheck %s

// Verify that `mlir.user` annotations survive the pre-backend canonicalizer in
// `torchdynamo-export-to-torch-backend-pipeline`. `torch.aten.to.other` is
// rewritten to `torch.aten.to.device` by the first canonicalizer pass, and then
// decomposed to `torch.aten.to.dtype` by `DecomposeComplexOps`.
// CHECK-LABEL:   func.func @test_aten_to_other_user_attrs_survive_pre_backend_canonicalizer(
// CHECK:           %[[RESULT:.*]] = torch.aten.to.dtype {{.*}} {mlir.user = [{my.tag = "to_other"}]}
// CHECK:           return %[[RESULT]]
func.func @test_aten_to_other_user_attrs_survive_pre_backend_canonicalizer(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f64>) -> !torch.vtensor<[2,3],f64> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.aten.to.other %arg0, %arg1, %false, %false, %none {mlir.user = [{my.tag = "to_other"}]} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f64>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,3],f64>
  return %0 : !torch.vtensor<[2,3],f64>
}
