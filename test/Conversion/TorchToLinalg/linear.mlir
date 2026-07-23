// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @test_matmul_user_attrs(
// CHECK:           linalg.matmul {mlir.user.tag = "layer_1"}
// CHECK-NOT:       internal.flag
func.func @test_matmul_user_attrs(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
  %0 = torch.aten.mm %arg0, %arg1 {mlir.user.tag = "layer_1", internal.flag = 42 : i64} : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}
