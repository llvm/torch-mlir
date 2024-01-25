// RUN: torch-mlir-opt %s | torch-mlir-opt | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK: #[[$ENCODING:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func.func @foo(
// CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[64,64],f32,#[[$ENCODING]]>)
// CHECK-NEXT:    return %[[A]] : !torch.vtensor<[64,64],f32,#[[$ENCODING]]>
func.func @foo(%arg0: !torch.vtensor<[64,64],f32,#CSR>) -> !torch.vtensor<[64,64],f32,#CSR> {
  return %arg0 : !torch.vtensor<[64,64],f32,#CSR>
}
