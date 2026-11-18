// RUN: torch-mlir-opt %s -split-input-file -convert-torch-onnx-to-torch | FileCheck %s --implicit-check-not=torch.operator

// Without an alpha attribute, alpha is 1.702.
// CHECK-LABEL: func.func @quick_gelu_default_alpha(
// CHECK-SAME: %[[X:.*]]: !torch.vtensor<[2,4],f32>
// CHECK: %[[ALPHA:.*]] = torch.constant.float 1.7020000219345093
// CHECK: %[[SCALED:.*]] = torch.aten.mul.Scalar %[[X]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[SIGMOID:.*]] = torch.aten.sigmoid %[[SCALED]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[Y:.*]] = torch.aten.mul.Tensor %[[X]], %[[SIGMOID]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: return %[[Y]]
func.func @quick_gelu_default_alpha(%x: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.QuickGelu"(%x) : (!torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// f16 is computed in f32 and rounded once at the end.
// CHECK-LABEL: func.func @quick_gelu_f16_alpha(
// CHECK-SAME: %[[X:.*]]: !torch.vtensor<[2,4],f16>
// CHECK: %[[X32:.*]] = torch.aten.to.dtype %[[X]], {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[ALPHA:.*]] = torch.constant.float 1.500000e+00
// CHECK: %[[SCALED:.*]] = torch.aten.mul.Scalar %[[X32]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[SIGMOID:.*]] = torch.aten.sigmoid %[[SCALED]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[Y32:.*]] = torch.aten.mul.Tensor %[[X32]], %[[SIGMOID]] : {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[Y:.*]] = torch.aten.to.dtype %[[Y32]], {{.*}} -> !torch.vtensor<[2,4],f16>
// CHECK: return %[[Y]]
func.func @quick_gelu_f16_alpha(%x: !torch.vtensor<[2,4],f16>) -> !torch.vtensor<[2,4],f16>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.QuickGelu"(%x) {torch.onnx.alpha = 1.500000e+00 : f32} : (!torch.vtensor<[2,4],f16>) -> !torch.vtensor<[2,4],f16>
  return %0 : !torch.vtensor<[2,4],f16>
}

// -----

// bf16 is computed in f32 as well.
// CHECK-LABEL: func.func @quick_gelu_bf16(
// CHECK-SAME: %[[X:.*]]: !torch.vtensor<[2,4],bf16>
// CHECK: %[[X32:.*]] = torch.aten.to.dtype %[[X]], {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[Y32:.*]] = torch.aten.mul.Tensor %[[X32]], {{.*}} -> !torch.vtensor<[2,4],f32>
// CHECK: %[[Y:.*]] = torch.aten.to.dtype %[[Y32]], {{.*}} -> !torch.vtensor<[2,4],bf16>
// CHECK: return %[[Y]]
func.func @quick_gelu_bf16(%x: !torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2,4],bf16>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %0 = torch.operator "onnx.QuickGelu"(%x) : (!torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2,4],bf16>
  return %0 : !torch.vtensor<[2,4],bf16>
}
