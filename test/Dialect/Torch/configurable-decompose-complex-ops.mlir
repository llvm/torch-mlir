// RUN: torch-mlir-opt -torch-decompose-complex-ops="skipOps=torch.aten.softmax.int,torch.aten.matmul" -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @matmul_no_decompose
// CHECK:            torch.aten.matmul %arg0, %arg1
func.func @matmul_no_decompose(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// ----
// CHECK-LABEL:   func.func @softmax_no_decompose
// CHECK:           torch.aten.softmax.int
func.func @softmax_no_decompose(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.softmax.int %arg0, %int1, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}