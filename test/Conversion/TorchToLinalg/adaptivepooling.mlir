// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @forward
func.func @forward(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,7],f32> {
  %int7 = torch.constant.int 7
  %output_size = torch.prim.ListConstruct %int7 : (!torch.int) -> !torch.list<int>
  %4 = torch.aten.adaptive_avg_pool1d %arg0, %output_size : !torch.vtensor<[?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,7],f32>
  return %4 : !torch.vtensor<[?,?,7],f32>
}