// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// -----

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func.func @sum(
// CHECK-SAME:  %[[A:.*]]: !torch.vtensor<[64,64],f32,#[[$CSR]]>) -> !torch.vtensor<[],f32>
// CHECK:       %[[S:.*]] = torch_c.to_builtin_tensor %[[A]] : !torch.vtensor<[64,64],f32,#[[$CSR]]> -> tensor<64x64xf32, #[[$CSR]]>
// CHECK:       linalg.generic {{{.*}}} ins(%[[S]] : tensor<64x64xf32, #[[$CSR]]>)
func.func @sum(%arg0: !torch.vtensor<[64,64],f32,#CSR>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none
     : !torch.vtensor<[64,64],f32,#CSR>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func.func @SpMM(
// CHECK-SAME:  %[[A:.*]]: !torch.vtensor<[8,16],f32,#[[$CSR]]>,
// CHECK-SAME:  %[[B:.*]]: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32>
// CHECK:       %[[S:.*]] = torch_c.to_builtin_tensor %[[A]] : !torch.vtensor<[8,16],f32,#[[$CSR]]> -> tensor<8x16xf32, #[[$CSR]]>
// CHECK:       %[[T:.*]] = torch_c.to_builtin_tensor %[[B]] : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
// CHECK:       linalg.matmul ins(%[[S]], %[[T]] : tensor<8x16xf32, #[[$CSR]]>, tensor<16x8xf32>)
func.func @SpMM(%arg0: !torch.vtensor<[8,16],f32,#CSR>,
                %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
  %0 = torch.aten.matmul %arg0, %arg1
     : !torch.vtensor<[8,16],f32,#CSR>,
       !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}
