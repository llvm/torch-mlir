// RUN: torch-mlir-opt <%s -pass-pipeline='builtin.module(torch-backend-to-stablehlo-backend-pipeline)' -split-input-file | FileCheck %s

// XFAIL: *
//
// Known gap: `convert-torch-to-stablehlo` forwards the annotation onto
// `chlo.broadcast_add`, but the upstream `chlo-legalize-to-stablehlo` pass
// that runs next in `createTorchBackendToStablehloBackendPipeline` rebuilds
// it as `stablehlo.add` without forwarding discardable attributes. Fixing this
// requires running `stablehlo::populateChloToStablehloPatterns` through a
// torch-mlir pass with the forwarding wrapper and listener installed. Remove
// the XFAIL once that lands. See docs/user_annotations.md.

// CHECK-LABEL:  func.func @linear_relu_user_attrs(
// CHECK-SAME:       %{{.*}}: tensor<1x4xf32> {mlir.user = {role = "input"}}
// CHECK:          stablehlo.add {{.*}} {mlir.user = [{lo = -1.000000e+00 : f64}]}
// CHECK-NOT:      mlir.user
// CHECK:          return
func.func @linear_relu_user_attrs(%arg0: !torch.vtensor<[1,4],f32> {mlir.user = {role = "input"}}, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[1,4],f32>
  %1 = torch.aten.add.Tensor %0, %arg2, %int1 {mlir.user = [{lo = -1.0 : f64}]} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[1,4],f32>
  %2 = torch.aten.relu %1 : !torch.vtensor<[1,4],f32> -> !torch.vtensor<[1,4],f32>
  return %2 : !torch.vtensor<[1,4],f32>
}
