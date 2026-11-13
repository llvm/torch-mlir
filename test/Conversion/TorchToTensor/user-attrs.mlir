// RUN: torch-mlir-opt <%s -convert-torch-to-tensor -split-input-file | FileCheck %s

// Verify that `mlir.user` annotations are forwarded onto the ops that replace
// annotated Torch ops lowered to the tensor dialect.

// CHECK-LABEL:  func.func @tensor_user_attrs(
// CHECK:          tensor.from_elements {{.*}} {mlir.user = [{tag = "tensor_tag"}]}
func.func @tensor_user_attrs(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[2],si64> {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %1 = torch.aten.tensor %0, %none, %none, %false {mlir.user = [{tag = "tensor_tag"}]} : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[2],si64>
  return %1 : !torch.vtensor<[2],si64>
}
