// RUN: torch-mlir-opt %s -convert-torch-to-linalg -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @gelu_none_f16(
// CHECK: %[[INPUT:.*]] = arith.extf %{{.*}} : f16 to f32
// CHECK: %[[ACTIVATION:.*]] = math.erf %{{.*}} : f32
// CHECK: %[[RESULT:.*]] = arith.mulf %[[INPUT]], %{{.*}} : f32
// CHECK: %[[OUTPUT:.*]] = arith.truncf %[[RESULT]] : f32 to f16
// CHECK: linalg.yield %[[OUTPUT]] : f16
func.func @gelu_none_f16(%input: !torch.vtensor<[7],f16>) -> !torch.vtensor<[7],f16> {
  %mode = torch.constant.str "none"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],f16>, !torch.str -> !torch.vtensor<[7],f16>
  return %result : !torch.vtensor<[7],f16>
}

// -----

// CHECK-LABEL: func.func @gelu_tanh_f16(
// CHECK: %[[INPUT:.*]] = arith.extf %{{.*}} : f16 to f32
// CHECK: %[[ACTIVATION:.*]] = math.tanh %{{.*}} : f32
// CHECK: %[[RESULT:.*]] = arith.mulf %[[INPUT]], %{{.*}} : f32
// CHECK: %[[OUTPUT:.*]] = arith.truncf %[[RESULT]] : f32 to f16
// CHECK: linalg.yield %[[OUTPUT]] : f16
func.func @gelu_tanh_f16(%input: !torch.vtensor<[7],f16>) -> !torch.vtensor<[7],f16> {
  %mode = torch.constant.str "tanh"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],f16>, !torch.str -> !torch.vtensor<[7],f16>
  return %result : !torch.vtensor<[7],f16>
}

// -----

// CHECK-LABEL: func.func @gelu_none_bf16(
// CHECK: %[[INPUT:.*]] = arith.extf %{{.*}} : bf16 to f32
// CHECK: %[[ACTIVATION:.*]] = math.erf %{{.*}} : f32
// CHECK: %[[RESULT:.*]] = arith.mulf %[[INPUT]], %{{.*}} : f32
// CHECK: %[[OUTPUT:.*]] = arith.truncf %[[RESULT]] : f32 to bf16
// CHECK: linalg.yield %[[OUTPUT]] : bf16
func.func @gelu_none_bf16(%input: !torch.vtensor<[7],bf16>) -> !torch.vtensor<[7],bf16> {
  %mode = torch.constant.str "none"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],bf16>, !torch.str -> !torch.vtensor<[7],bf16>
  return %result : !torch.vtensor<[7],bf16>
}

// -----

// CHECK-LABEL: func.func @gelu_tanh_bf16(
// CHECK: %[[INPUT:.*]] = arith.extf %{{.*}} : bf16 to f32
// CHECK: %[[ACTIVATION:.*]] = math.tanh %{{.*}} : f32
// CHECK: %[[RESULT:.*]] = arith.mulf %[[INPUT]], %{{.*}} : f32
// CHECK: %[[OUTPUT:.*]] = arith.truncf %[[RESULT]] : f32 to bf16
// CHECK: linalg.yield %[[OUTPUT]] : bf16
func.func @gelu_tanh_bf16(%input: !torch.vtensor<[7],bf16>) -> !torch.vtensor<[7],bf16> {
  %mode = torch.constant.str "tanh"
  %result = torch.aten.gelu %input, %mode : !torch.vtensor<[7],bf16>, !torch.str -> !torch.vtensor<[7],bf16>
  return %result : !torch.vtensor<[7],bf16>
}
