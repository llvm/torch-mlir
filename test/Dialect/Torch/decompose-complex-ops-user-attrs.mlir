// RUN: torch-mlir-opt <%s -torch-decompose-complex-ops -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_linear_user_attrs(
// With the new per-result forwarding, only the final operation gets the attributes
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.transpose.int %arg1
// CHECK-NOT:       mlir.user
// CHECK:           %[[MATMUL:.*]] = torch.aten.mm %arg0, %[[TRANSPOSE]]
// CHECK-NOT:       mlir.user
// CHECK:           %[[RESULT:.*]] = torch.aten.add.Tensor %[[MATMUL]], %arg2, %{{.*}} {mlir.user = [{my.range_hi = 1.000000e+00 : f64, my.range_lo = -1.000000e+00 : f64}]}
// CHECK:           return %[[RESULT]]
func.func @test_linear_user_attrs(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 {mlir.user = [{my.range_hi = 1.0 : f64, my.range_lo = -1.0 : f64}]} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// CHECK-LABEL:   func.func @test_linear_no_bias_user_attrs(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.transpose.int %arg1
// CHECK-NOT:       mlir.user
// CHECK:           %[[MATMUL:.*]] = torch.aten.mm %arg0, %[[TRANSPOSE]] {mlir.user = [{tag = "layer_1"}]}
// CHECK:           return %[[MATMUL]]
func.func @test_linear_no_bias_user_attrs(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[1,4],f32> {
  %none = torch.constant.none
  %0 = torch.aten.linear %arg0, %arg1, %none {mlir.user = [{tag = "layer_1"}]} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.none -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// Test that internal (non-user) attributes are not forwarded
// CHECK-LABEL:   func.func @test_linear_internal_attrs_not_forwarded(
// CHECK:           %[[TRANSPOSE:.*]] = torch.aten.transpose.int %arg1
// CHECK-NOT:       mlir.user
// CHECK-NOT:       internal.flag
// CHECK:           %[[MATMUL:.*]] = torch.aten.mm %arg0, %[[TRANSPOSE]]
// CHECK-NOT:       mlir.user
// CHECK-NOT:       internal.flag
// CHECK:           %[[RESULT:.*]] = torch.aten.add.Tensor %[[MATMUL]], %arg2, %{{.*}} {mlir.user = [{tag = "public"}]}
// CHECK-NOT:       internal.flag
// CHECK:           return %[[RESULT]]
func.func @test_linear_internal_attrs_not_forwarded(%arg0: !torch.vtensor<[1,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 {mlir.user = [{tag = "public"}], internal.flag = 42 : i64} : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
  return %0 : !torch.vtensor<[1,4],f32>
}

// -----

// Test multi-result operation with different annotations per result
// This demonstrates that per-result attributes are properly represented and forwarded
// topk decomposes into sort + slice, with attributes forwarded to each slice
// CHECK-LABEL:   func.func @test_multi_result_different_attrs(
// CHECK:           torch.aten.sort
// CHECK:           torch.aten.slice.Tensor{{.*}}{mlir.user = [{my.range_lo = -1.000000e+00 : f64, my.tag = "values"}]}
// CHECK:           torch.aten.slice.Tensor{{.*}}{mlir.user = [{my.tag = "indices", my.type = "int"}]}
func.func @test_multi_result_different_attrs(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>) {
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.topk %arg0, %int2, %int-1, %true, %true {mlir.user = [{my.tag = "values", my.range_lo = -1.0 : f64}, {my.tag = "indices", my.type = "int"}]} : !torch.vtensor<[4],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>
  return %values, %indices : !torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>
}

// -----

// Regression test: a decomposition may drop an unused result by passing a null
// Value to `rewriter.replaceOp`, e.g.
// `rewriter.replaceOp(op, {maxPool.getResult(), Value()})`. The attribute
// forwarding listener must skip null replacements rather than calling
// `getDefiningOp()` on them (which asserts inside `dyn_cast<OpResult>`).
// Here the `indices` result of `aten.adaptive_max_pool1d` is unused.
// CHECK-LABEL:   func.func @test_null_replacement_for_unused_result(
// CHECK:           %[[POOL:.*]] = torch.aten.max_pool1d {{.*}} {mlir.user = [{my.tag = "values"}]}
// CHECK:           return %[[POOL]]
func.func @test_null_replacement_for_unused_result(%arg0: !torch.vtensor<[1,512,7],f32>) -> !torch.vtensor<[1,512,1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %values, %indices = torch.aten.adaptive_max_pool1d %arg0, %0 {mlir.user = [{my.tag = "values"}, {my.tag = "indices"}]} : !torch.vtensor<[1,512,7],f32>, !torch.list<int> -> !torch.vtensor<[1,512,1],f32>, !torch.vtensor<[1,512,1],si64>
  return %values : !torch.vtensor<[1,512,1],f32>
}
