// RUN: torch-mlir-opt -torch-decompose-complex-ops="legal-ops=aten.softmax.int,aten.sinc" -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.softmax.int$cst_dim
func.func @torch.aten.softmax.int$cst_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  // CHECK: torch.aten.softmax.int
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.sinc$legal
func.func @torch.aten.sinc$legal(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
  // CHECK: torch.aten.sinc
  // CHECK-NOT: torch.aten.where.self
  %0 = torch.aten.sinc %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}
