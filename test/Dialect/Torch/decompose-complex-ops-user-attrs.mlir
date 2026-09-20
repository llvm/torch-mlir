// RUN: torch-mlir-opt <%s -torch-decompose-complex-ops -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_linear_user_attrs(
// With the new per-result forwarding, only the final operation gets the attributes
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.t %arg1
// CHECK-NOT:       mlir.user
// CHECK:           %[[MATMUL:.*]] = torch.aten.matmul %arg0, %[[TRANSPOSE]]
// CHECK-NOT:       mlir.user
// CHECK:           %[[RESULT:.*]] = torch.aten.add.Tensor %[[MATMUL]], %arg2, %{{.*}} {mlir.user.my.range_hi = 1.000000e+00 : f64, mlir.user.my.range_lo = -1.000000e+00 : f64}
// CHECK:           return %[[RESULT]]
func.func @test_linear_user_attrs(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 {mlir.user = [{my.range_hi = 1.0 : f64, my.range_lo = -1.0 : f64}]} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// CHECK-LABEL:   func.func @test_linear_no_bias_user_attrs(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.t %arg1
// CHECK-NOT:       mlir.user
// CHECK:           %[[MATMUL:.*]] = torch.aten.matmul %arg0, %[[TRANSPOSE]] {mlir.user.tag = "layer_1"}
// CHECK:           return %[[MATMUL]]
func.func @test_linear_no_bias_user_attrs(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[1,4],f32> {
  %none = torch.constant.none
  %0 = torch.aten.linear %arg0, %arg1, %none {mlir.user = [{tag = "layer_1"}]} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.none -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// Test that internal (non-user) attributes are not forwarded
// CHECK-LABEL:   func.func @test_linear_internal_attrs_not_forwarded(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.t %arg1
// CHECK-NOT:       mlir.user
// CHECK-NOT:       internal.flag
// CHECK:           %[[MATMUL:.*]] = torch.aten.matmul %arg0, %[[TRANSPOSE]]
// CHECK-NOT:       mlir.user
// CHECK-NOT:       internal.flag
// CHECK:           %[[RESULT:.*]] = torch.aten.add.Tensor %[[MATMUL]], %arg2, %{{.*}} {mlir.user.tag = "public"}
// CHECK-NOT:       internal.flag
// CHECK:           return %[[RESULT]]
func.func @test_linear_internal_attrs_not_forwarded(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 {mlir.user = [{tag = "public"}], internal.flag = 42 : i64} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// Test multi-result operation with different annotations per result
// This demonstrates that per-result attributes are properly represented and forwarded
// CHECK-LABEL:   func.func @test_multi_result_different_attrs(
// CHECK:           %[[VALUES:.*]], %[[INDICES:.*]] = torch.aten.topk
// CHECK-SAME:        {mlir.user = [{my.range_lo = -1.000000e+00 : f64, my.tag = "values"}, {my.tag = "indices", my.type = "int"}]}
// CHECK:           return %[[VALUES]], %[[INDICES]]
func.func @test_multi_result_different_attrs(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>) {
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.topk %arg0, %int2, %int-1, %true, %true {mlir.user = [{my.tag = "values", my.range_lo = -1.0 : f64}, {my.tag = "indices", my.type = "int"}]} : !torch.vtensor<[4],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>
  return %values, %indices : !torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>
}
