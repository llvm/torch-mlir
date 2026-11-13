// RUN: torch-mlir-opt <%s -convert-torch-to-arith -split-input-file | FileCheck %s

// Verify that `mlir.user` annotations are forwarded onto the ops that replace
// annotated Torch scalar ops, including intermediate ops whose results only
// feed other converted ops.

// CHECK-LABEL:  func.func @scalar_user_attrs(
// CHECK:          arith.addi {{.*}} {mlir.user = [{tag = "add_tag"}]}
// CHECK:          arith.muli {{.*}} {mlir.user = [{tag = "mul_tag"}]}
func.func @scalar_user_attrs(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.add.int %arg0, %arg1 {mlir.user = [{tag = "add_tag"}]} : !torch.int, !torch.int -> !torch.int
  %1 = torch.aten.mul.int %0, %arg1 {mlir.user = [{tag = "mul_tag"}]} : !torch.int, !torch.int -> !torch.int
  return %1 : !torch.int
}
