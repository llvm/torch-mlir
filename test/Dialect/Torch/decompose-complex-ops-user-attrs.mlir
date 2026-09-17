// RUN: torch-mlir-opt <%s -torch-decompose-complex-ops -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_linear_user_attrs(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.t %arg1 {mlir.user.my.range_hi = 1.000000e+00 : f64, mlir.user.my.range_lo = -1.000000e+00 : f64}
// CHECK:           %[[MATMUL:.*]] = torch.aten.matmul %arg0, %[[TRANSPOSE]] {mlir.user.my.range_hi = 1.000000e+00 : f64, mlir.user.my.range_lo = -1.000000e+00 : f64}
// CHECK:           %[[RESULT:.*]] = torch.aten.add.Tensor %[[MATMUL]], %arg2, %{{.*}} {mlir.user.my.range_hi = 1.000000e+00 : f64, mlir.user.my.range_lo = -1.000000e+00 : f64}
// CHECK:           return %[[RESULT]]
func.func @test_linear_user_attrs(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 {mlir.user.my.range_lo = -1.0 : f64, mlir.user.my.range_hi = 1.0 : f64} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// CHECK-LABEL:   func.func @test_linear_no_bias_user_attrs(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.t %arg1 {mlir.user.tag = "layer_1"}
// CHECK:           %[[MATMUL:.*]] = torch.aten.matmul %arg0, %[[TRANSPOSE]] {mlir.user.tag = "layer_1"}
// CHECK:           return %[[MATMUL]]
func.func @test_linear_no_bias_user_attrs(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[1,4],f32> {
  %none = torch.constant.none
  %0 = torch.aten.linear %arg0, %arg1, %none {mlir.user.tag = "layer_1"} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.none -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// Test that internal (non-user) attributes are not forwarded
// CHECK-LABEL:   func.func @test_linear_internal_attrs_not_forwarded(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.t %arg1 {mlir.user.tag = "public"}
// CHECK-NOT:       internal.flag
// CHECK:           %[[MATMUL:.*]] = torch.aten.matmul %arg0, %[[TRANSPOSE]] {mlir.user.tag = "public"}
// CHECK-NOT:       internal.flag
// CHECK:           %[[RESULT:.*]] = torch.aten.add.Tensor %[[MATMUL]], %arg2, %{{.*}} {mlir.user.tag = "public"}
// CHECK-NOT:       internal.flag
// CHECK:           return %[[RESULT]]
func.func @test_linear_internal_attrs_not_forwarded(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 {mlir.user.tag = "public", internal.flag = 42 : i64} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}
