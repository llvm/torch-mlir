// RUN: torch-mlir-opt <%s -convert-torch-to-tcp -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>} : tensor<4xf32>
// CHECK:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:         return %[[T2]] : !torch.vtensor<[4],f32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],f32> {
  %0 = torch.vtensor.literal(dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],si32> {
//       CHECK:  %[[T1:.+]] = tcp.const {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : tensor<4xi32>
//       CHECK:  %[[T2:.+]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xi32> -> !torch.vtensor<[4],si32>
//       CHECK:  return %[[T2]] : !torch.vtensor<[4],si32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],si32> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi32>) : !torch.vtensor<[4],si32>
  return %0 : !torch.vtensor<[4],si32>
}

// -----

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],ui8> {
//       CHECK:  %[[T1:.+]] = tcp.const {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
//       CHECK:  %[[T2:.+]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xi8> -> !torch.vtensor<[4],ui8>
//       CHECK:  return %[[T2]] : !torch.vtensor<[4],ui8>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],ui8> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  return %0 : !torch.vtensor<[4],ui8>
}
