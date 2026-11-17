// RUN: torch-mlir-opt %s -torch-restructure-non-constant-axes | FileCheck %s

// Verify the pass rewrites the upstream single-element
// `item(select.int(%arg1, 0, 0))` chain into a direct `item(%arg1)`,
// leaving no orphan select.int or constant.int behind.

// CHECK-LABEL: func.func @sum_runtime_single_axis
// CHECK-NOT: torch.aten.select.int
// CHECK: torch.aten.item %arg1
// CHECK: torch.aten.sum.dim_IntList
func.func @sum_runtime_single_axis(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> {
  %none = torch.constant.none
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %0 = torch.aten.select.int %arg1, %int0, %int0 : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
  %1 = torch.aten.item %0 : !torch.vtensor<[1],si64> -> !torch.int
  %2 = torch.prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
  %3 = torch.aten.sum.dim_IntList %arg0, %2, %true, %none : !torch.vtensor<[3,2,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1,2],f32>
  return %3 : !torch.vtensor<[3,1,2],f32>
}
