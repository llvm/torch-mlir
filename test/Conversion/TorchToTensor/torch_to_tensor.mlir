// RUN: torch-mlir-opt <%s -convert-torch-to-tensor | FileCheck %s

// CHECK-LABEL: func.func @test_shape
func.func @test_shape(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3],si64> {
  // CHECK: %[[SHAPE:.+]] = arith.constant dense<[3, 4, 5]> : tensor<3xi64>
  %0 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}
