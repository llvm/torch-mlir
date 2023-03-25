// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-function-to-torch-backend-pipeline{backend-legal-ops=aten.square,aten.argmax})' -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.square
func.func @torch.aten.square(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  // CHECK: torch.aten.square
  %0 = torch.aten.square %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL: func.func @torch.aten.argmax
func.func @torch.aten.argmax(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[1,?],si64> {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  // CHECK: torch.aten.argmax
  %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,?],si64>
  return %0 : !torch.vtensor<[1,?],si64>
}
