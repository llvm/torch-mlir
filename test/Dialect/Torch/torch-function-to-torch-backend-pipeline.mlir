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

// CHECK-LABEL: func.func @torch.uint8
func.func @torch.uint8(%arg0: !torch.tensor {torch.type_bound = !torch.vtensor<[3,4],ui8>}) -> !torch.tensor {
  %int12 = torch.constant.int 12
  %0 = torch.prim.ListConstruct %int12 : (!torch.int) -> !torch.list<int>
  // CHECK: torch.aten.view
  // CHECK-SAME: !torch.vtensor<[12],ui8>
  %1 = torch.aten.reshape %arg0, %0 : !torch.tensor, !torch.list<int> -> !torch.tensor
  return %1 : !torch.tensor
}

// CHECK-LABEL: func.func @torch.f8type
func.func @torch.f8type(%arg0: !torch.vtensor<[5,3],f8E4M3FNUZ>) -> !torch.vtensor<[5,3],f8E4M3FNUZ> {
  // CHECK: torch.aten.exp
  // CHECK: torch.aten.log1p
  // CHECK: torch.aten.tanh
  // CHECK-SAME: !torch.vtensor<[5,3],f8E4M3FNUZ>
  %0 = torch.aten.mish %arg0 : !torch.vtensor<[5,3],f8E4M3FNUZ> -> !torch.vtensor<[5,3],f8E4M3FNUZ>
  return %0 : !torch.vtensor<[5,3],f8E4M3FNUZ>
}
