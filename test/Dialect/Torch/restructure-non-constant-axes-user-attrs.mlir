// RUN: torch-mlir-opt %s -torch-restructure-non-constant-axes | FileCheck %s

// Verify that `mlir.user` annotations on a reduction with a non-constant dim
// are forwarded onto the value that replaces it (the final reshape back to
// the original result shape). Note that the pattern also copies all of the
// original op's attributes onto the restructured reduction.

// CHECK-LABEL: func.func @sum_user_attrs(
// CHECK:         torch.aten.sum.dim_IntList {{.*}} {mlir.user = [{tag = "sum_tag"}]}
// CHECK:         torch.aten.view {{.*}} {mlir.user = [{tag = "sum_tag"}]}
func.func @sum_user_attrs(%arg0: !torch.vtensor<[2,3,4],f32>, %dim: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %dims = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
  %0 = torch.aten.sum.dim_IntList %arg0, %dims, %true, %none {mlir.user = [{tag = "sum_tag"}]} : !torch.vtensor<[2,3,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}
