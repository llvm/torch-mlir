// RUN: torch-mlir-opt %s -convert-torch-to-stablehlo -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @gelu_none_f16(
// CHECK: %[[INPUT:.*]] = stablehlo.convert %{{.*}} : (tensor<7xf16>) -> tensor<7xf32>
// CHECK: chlo.erf %{{.*}} : tensor<7xf32> -> tensor<7xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.multiply %{{.*}} : tensor<7xf32>
// CHECK: stablehlo.convert %[[RESULT]] : (tensor<7xf32>) -> tensor<7xf16>
func.func @gelu_none_f16(%input: !torch.vtensor<[7],f16>) -> !torch.vtensor<[7],f16> {
  %mode = torch.constant.str "none"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],f16>, !torch.str -> !torch.vtensor<[7],f16>
  return %result : !torch.vtensor<[7],f16>
}

// -----

// CHECK-LABEL: func.func @gelu_tanh_f16(
// CHECK: %[[INPUT:.*]] = stablehlo.convert %{{.*}} : (tensor<7xf16>) -> tensor<7xf32>
// CHECK: stablehlo.tanh %{{.*}} : tensor<7xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.multiply %{{.*}} : tensor<7xf32>
// CHECK: stablehlo.convert %[[RESULT]] : (tensor<7xf32>) -> tensor<7xf16>
func.func @gelu_tanh_f16(%input: !torch.vtensor<[7],f16>) -> !torch.vtensor<[7],f16> {
  %mode = torch.constant.str "tanh"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],f16>, !torch.str -> !torch.vtensor<[7],f16>
  return %result : !torch.vtensor<[7],f16>
}

// -----

// CHECK-LABEL: func.func @gelu_none_bf16(
// CHECK: %[[INPUT:.*]] = stablehlo.convert %{{.*}} : (tensor<7xbf16>) -> tensor<7xf32>
// CHECK: chlo.erf %{{.*}} : tensor<7xf32> -> tensor<7xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.multiply %{{.*}} : tensor<7xf32>
// CHECK: stablehlo.convert %[[RESULT]] : (tensor<7xf32>) -> tensor<7xbf16>
func.func @gelu_none_bf16(%input: !torch.vtensor<[7],bf16>) -> !torch.vtensor<[7],bf16> {
  %mode = torch.constant.str "none"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],bf16>, !torch.str -> !torch.vtensor<[7],bf16>
  return %result : !torch.vtensor<[7],bf16>
}

// -----

// CHECK-LABEL: func.func @gelu_tanh_bf16(
// CHECK: %[[INPUT:.*]] = stablehlo.convert %{{.*}} : (tensor<7xbf16>) -> tensor<7xf32>
// CHECK: stablehlo.tanh %{{.*}} : tensor<7xf32>
// CHECK: %[[RESULT:.*]] = stablehlo.multiply %{{.*}} : tensor<7xf32>
// CHECK: stablehlo.convert %[[RESULT]] : (tensor<7xf32>) -> tensor<7xbf16>
func.func @gelu_tanh_bf16(%input: !torch.vtensor<[7],bf16>) -> !torch.vtensor<[7],bf16> {
  %mode = torch.constant.str "tanh"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],bf16>, !torch.str -> !torch.vtensor<[7],bf16>
  return %result : !torch.vtensor<[7],bf16>
}
