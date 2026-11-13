// RUN: torch-mlir-opt <%s -split-input-file | FileCheck %s

// This test file demonstrates the DESIRED behavior for per-result annotations.
// Currently, MLIR operation attributes apply to the whole operation, not individual results.
// The proposed solution is to use result attributes (if available) or encode result index
// in the attribute name.

// CHECK-LABEL: func.func @test_topk_different_result_annotations
// DESIRED: Each result should preserve its own annotations
// Option 1: Use result attributes (if MLIR Python bindings support them)
// %values, %indices = torch.aten.topk ...
//   {mlir.user.result_attrs = [{my.tag = "values", my.range_lo = -1.0}, {my.tag = "indices"}]}
//
// Option 2: Encode result index in attribute name
// %values, %indices = torch.aten.topk ...
//   {mlir.user.0.my.tag = "values", mlir.user.0.my.range_lo = -1.0,
//    mlir.user.1.my.tag = "indices"}
func.func @test_topk_different_result_annotations(%arg0: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>) {
  %int2 = torch.constant.int 2
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true

  // Hypothetical: if we could annotate individual results
  // %values {mlir.user.my.tag = "values", mlir.user.my.range_lo = -1.0}
  // %indices {mlir.user.my.tag = "indices"}
  %values, %indices = torch.aten.topk %arg0, %int2, %int-1, %true, %true : !torch.vtensor<[4],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>

  return %values, %indices : !torch.vtensor<[2],f32>, !torch.vtensor<[2],si64>
}

// -----

// This test shows the over-propagation problem in decomposition
// CHECK-LABEL: func.func @test_forwarding_to_corresponding_replacement_only
func.func @test_forwarding_to_corresponding_replacement_only(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
  // Suppose this op has user annotation and gets decomposed into multiple ops
  // %0 = torch.some.op %arg0 {mlir.user.0.my.tag = "result0", mlir.user.1.my.tag = "result1"}
  //
  // After decomposition:
  // %helper = torch.helper.op %arg0        // Should NOT get annotations (helper)
  // %intermediate = torch.compute %helper  // Should get mlir.user.0.* annotations (defines result 0)
  // %final = torch.finalize %intermediate  // Should get mlir.user.1.* annotations (defines result 1)
  //
  // Current ForwardingListener: Copies ALL annotations to ALL three ops (WRONG)
  // Desired: Copy only the annotations for the result being replaced to the replacement op

  %0 = torch.aten.relu %arg0 : !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}
