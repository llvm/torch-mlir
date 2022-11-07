// RUN: torch-mlir-opt <%s -convert-torch-backend-legal-to-tosa-custom="custom-ops=torch.aten._softmax" -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten._softmax(
// CHECK-SAME:      %[[VAL_0:.*]]: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int -1
// CHECK:           %[[VAL_3:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
// CHECK:           %[[VAL_5:.*]] = "tosa.const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
// CHECK:           %[[VAL_6:.*]] = "tosa.custom"(%[[VAL_1]], %[[VAL_4]], %[[VAL_5]]) {config = "torch_mlir", identifier = "aten._softmax"} : (tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<4x4xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x4xf32> -> !torch.vtensor<[4,4],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[4,4],f32>
// CHECK:         }
func.func @torch.aten._softmax(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %0 = torch.aten._softmax %arg0, %int-1, %false : !torch.vtensor<[4,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}
