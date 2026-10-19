// RUN: torch-mlir-opt %s -split-input-file -convert-torch-onnx-to-torch | FileCheck %s --implicit-check-not=torch.operator

// CHECK-LABEL: @fast_gelu_no_bias(
func.func @fast_gelu_no_bias(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %arg0, %[[TANH]] : !torch.vtensor<[2,4],f32>, !torch.str -> !torch.vtensor<[2,4],f32>
  // CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.FastGelu"(%input) : (!torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: @fast_gelu_none_bias(
func.func @fast_gelu_none_bias(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  %none = torch.constant.none
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %arg0, %[[TANH]] : !torch.vtensor<[2,4],f32>, !torch.str -> !torch.vtensor<[2,4],f32>
  // CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[2,4],f32>
  %0 = torch.operator "onnx.FastGelu"(%input, %none) : (!torch.vtensor<[2,4],f32>, !torch.none) -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// Bias must be broadcast and added before applying the activation.
// CHECK-LABEL: @fast_gelu_bias(
func.func @fast_gelu_bias(%input: !torch.vtensor<[2,3,4],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,3,4],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-NEXT: %[[BIASED:.+]] = torch.aten.add.Tensor %arg0, %arg1, %[[ONE]] : !torch.vtensor<[2,3,4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[2,3,4],f32>
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %[[BIASED]], %[[TANH]] : !torch.vtensor<[2,3,4],f32>, !torch.str -> !torch.vtensor<[2,3,4],f32>
  // CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[2,3,4],f32>
  %0 = torch.operator "onnx.FastGelu"(%input, %bias) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,3,4],f32>
  return %0 : !torch.vtensor<[2,3,4],f32>
}

// -----

// CHECK-LABEL: @fast_gelu_dynamic_bias(
func.func @fast_gelu_dynamic_bias(%input: !torch.vtensor<[?,?,?],f32>, %bias: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-NEXT: %[[BIASED:.+]] = torch.aten.add.Tensor %arg0, %arg1, %[[ONE]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?,?,?],f32>
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %[[BIASED]], %[[TANH]] : !torch.vtensor<[?,?,?],f32>, !torch.str -> !torch.vtensor<[?,?,?],f32>
  // CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[?,?,?],f32>
  %0 = torch.operator "onnx.FastGelu"(%input, %bias) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// -----

// CHECK-LABEL: @fast_gelu_f16_bias(
func.func @fast_gelu_f16_bias(%input: !torch.vtensor<[2,4],f16>, %bias: !torch.vtensor<[4],f16>) -> !torch.vtensor<[2,4],f16>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[INPUT:.+]] = torch.aten.to.dtype %arg0, %{{.*}} : !torch.vtensor<[2,4],f16>, {{.*}} -> !torch.vtensor<[2,4],f32>
  // CHECK: %[[BIAS:.+]] = torch.aten.to.dtype %arg1, %{{.*}} : !torch.vtensor<[4],f16>, {{.*}} -> !torch.vtensor<[4],f32>
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-NEXT: %[[BIASED:.+]] = torch.aten.add.Tensor %[[INPUT]], %[[BIAS]], %[[ONE]] : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %[[BIASED]], %[[TANH]] : !torch.vtensor<[2,4],f32>, !torch.str -> !torch.vtensor<[2,4],f32>
  // CHECK: %[[OUTPUT:.+]] = torch.aten.to.dtype %[[RESULT]], %{{.*}} : !torch.vtensor<[2,4],f32>, {{.*}} -> !torch.vtensor<[2,4],f16>
  // CHECK-NEXT: return %[[OUTPUT]] : !torch.vtensor<[2,4],f16>
  %0 = torch.operator "onnx.FastGelu"(%input, %bias) : (!torch.vtensor<[2,4],f16>, !torch.vtensor<[4],f16>) -> !torch.vtensor<[2,4],f16>
  return %0 : !torch.vtensor<[2,4],f16>
}

// -----

// CHECK-LABEL: @fast_gelu_bf16_bias(
func.func @fast_gelu_bf16_bias(%input: !torch.vtensor<[2,4],bf16>, %bias: !torch.vtensor<[4],bf16>) -> !torch.vtensor<[2,4],bf16>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK: %[[INPUT:.+]] = torch.aten.to.dtype %arg0, %{{.*}} : !torch.vtensor<[2,4],bf16>, {{.*}} -> !torch.vtensor<[2,4],f32>
  // CHECK: %[[BIAS:.+]] = torch.aten.to.dtype %arg1, %{{.*}} : !torch.vtensor<[4],bf16>, {{.*}} -> !torch.vtensor<[4],f32>
  // CHECK: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-NEXT: %[[BIASED:.+]] = torch.aten.add.Tensor %[[INPUT]], %[[BIAS]], %[[ONE]] : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %[[BIASED]], %[[TANH]] : !torch.vtensor<[2,4],f32>, !torch.str -> !torch.vtensor<[2,4],f32>
  // CHECK: %[[OUTPUT:.+]] = torch.aten.to.dtype %[[RESULT]], %{{.*}} : !torch.vtensor<[2,4],f32>, {{.*}} -> !torch.vtensor<[2,4],bf16>
  // CHECK-NEXT: return %[[OUTPUT]] : !torch.vtensor<[2,4],bf16>
  %0 = torch.operator "onnx.FastGelu"(%input, %bias) : (!torch.vtensor<[2,4],bf16>, !torch.vtensor<[4],bf16>) -> !torch.vtensor<[2,4],bf16>
  return %0 : !torch.vtensor<[2,4],bf16>
}

// -----

// CHECK-LABEL: @fast_gelu_bf16(
func.func @fast_gelu_bf16(%input: !torch.vtensor<[4],bf16>) -> !torch.vtensor<[4],bf16>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %arg0, %[[TANH]] : !torch.vtensor<[4],bf16>, !torch.str -> !torch.vtensor<[4],bf16>
  // CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[4],bf16>
  %0 = torch.operator "onnx.FastGelu"(%input) : (!torch.vtensor<[4],bf16>) -> !torch.vtensor<[4],bf16>
  return %0 : !torch.vtensor<[4],bf16>
}

// -----

// CHECK-LABEL: @fast_gelu_f64_scalar(
func.func @fast_gelu_f64_scalar(%input: !torch.vtensor<[],f64>) -> !torch.vtensor<[],f64>
    attributes {torch.onnx_meta.opset_version = 17 : si64,
                torch.onnx_meta.opset_versions = {"com.microsoft" = 1 : si64}} {
  // CHECK-NOT: torch.aten.add.Tensor
  // CHECK: %[[TANH:.+]] = torch.constant.str "tanh"
  // CHECK-NEXT: %[[RESULT:.+]] = torch.aten.gelu %arg0, %[[TANH]] : !torch.vtensor<[],f64>, !torch.str -> !torch.vtensor<[],f64>
  // CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[],f64>
  %0 = torch.operator "onnx.FastGelu"(%input) : (!torch.vtensor<[],f64>) -> !torch.vtensor<[],f64>
  return %0 : !torch.vtensor<[],f64>
}
