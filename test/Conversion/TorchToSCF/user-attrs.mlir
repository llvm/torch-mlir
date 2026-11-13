// RUN: torch-mlir-opt <%s -convert-torch-to-scf -split-input-file | FileCheck %s

// Verify that `mlir.user` annotations are forwarded onto the ops that replace
// annotated Torch control-flow ops.

// CHECK-LABEL:  func.func @prim_if_user_attrs(
// CHECK:          scf.if
// CHECK:          } {mlir.user = [{tag = "if_tag"}]}
func.func @prim_if_user_attrs(%arg0: !torch.bool, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.prim.If %arg0 -> (!torch.vtensor<[2,3],f32>) {
    torch.prim.If.yield %arg1 : !torch.vtensor<[2,3],f32>
  } else {
    torch.prim.If.yield %arg2 : !torch.vtensor<[2,3],f32>
  } {mlir.user = [{tag = "if_tag"}]}
  return %0 : !torch.vtensor<[2,3],f32>
}
