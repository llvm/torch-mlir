// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)'  -split-input-file -verify-diagnostics | FileCheck %s

// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline{allow-non-finites=false})'  -split-input-file -verify-diagnostics | FileCheck %s -check-prefix=UNSUPPORTED-NON-FINITES

// -----
// COM: this test only locks down that allow-non-finites config produces inf/realmax correctly
// CHECK-LABEL: func.func @torch.aten.min.dim$basic
// CHECK: arith.constant 0x7F800000 : f32
// CHECK-NOT: arith.constant 3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES: arith.constant 3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES-NOT: arith.constant 0x7F800000 : f32
func.func @torch.aten.min.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.min.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}

// -----
// COM: this test only locks down that allow-non-finites config produces inf/realmax correctly
// CHECK-LABEL: func.func @torch.aten.max.dim$basic
// CHECK: arith.constant 0xFF800000 : f32
// CHECK-NOT: arith.constant -3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES: arith.constant -3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES-NOT: arith.constant 0xFF800000 : f32
func.func @torch.aten.max.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.max.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}
