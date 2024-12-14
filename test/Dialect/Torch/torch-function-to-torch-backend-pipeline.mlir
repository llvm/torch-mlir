// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-function-to-torch-backend-pipeline{backend-legal-ops=aten.square,aten.argmax,torch.aten.round.decimals})' -split-input-file %s | FileCheck %s

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

// Test that "torch.aten.round.decimals" was considered legal after explicitly specifying it in pass options.
// CHECK-LABEL: func.func @torch_aten_round_decimals
func.func @torch_aten_round_decimals(%0: !torch.vtensor<[1,1024,1024,3],f32>) -> !torch.vtensor<[1, 1024,1024,3],f32> {
  %int0 = torch.constant.int 0
  %1 = torch.operator "torch.aten.round.decimals"(%0, %int0) : (!torch.vtensor<[1,1024,1024,3],f32>, !torch.int) -> !torch.vtensor<[1,1024,1024,3],f32>
  return %1 : !torch.vtensor<[1, 1024,1024,3],f32>
}
