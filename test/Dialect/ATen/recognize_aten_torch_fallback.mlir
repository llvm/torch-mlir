// RUN: npcomp-opt %s -aten-recognize-torch-fallback -split-input-file |& FileCheck %s
// Note that this test is not exhaustive with respect to ops (since the facility
// is generic). Instead, it uses examplar ops to test various types of
// conversions.

// CHECK-LABEL: func @graph
func @graph(%arg0: !torch.nn.Module<"__torch__.torch.nn.modules.conv.Conv2d">, %arg1: !numpy.ndarray<*:!numpy.any_dtype>, %arg2: !numpy.ndarray<*:!numpy.any_dtype>, %arg3: !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  // CHECK: %[[RESULTS:.*]] = refback.torch_fallback %arg1, %arg2, %arg3, %0, %1, %2, %c1_i64 -> (!numpy.ndarray<*:!numpy.any_dtype>) {
  // CHECK: %[[STRIDE:.*]] = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  // CHECK: %[[PADDING:.*]] = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  // CHECK: %[[DILATION:.*]] = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  // CHECK: %[[KERNEL_CALL:.*]] = torch.kernel_call "aten::conv2d" %arg1, %arg2, %arg3, %[[STRIDE]], %[[PADDING]], %[[DILATION]], %c1_i64 : (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> !numpy.ndarray<*:!numpy.any_dtype> {sigArgTypes = ["Tensor", "Tensor", "Tensor?", "int[]", "int[]", "int[]", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
  // CHECK: refback.torch_fallback_yield %[[KERNEL_CALL]] : !numpy.ndarray<*:!numpy.any_dtype>
  // CHECK: return %[[RESULTS]] : !numpy.ndarray<*:!numpy.any_dtype>
  %c0_i64 = constant 0 : i64
  %c1_i64 = constant 1 : i64
  %0 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %1 = basicpy.build_list %c0_i64, %c0_i64 : (i64, i64) -> !basicpy.ListType
  %2 = basicpy.build_list %c1_i64, %c1_i64 : (i64, i64) -> !basicpy.ListType
  %3 = torch.kernel_call "aten::conv2d" %arg1, %arg2, %arg3, %0, %1, %2, %c1_i64 : (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> !numpy.ndarray<*:!numpy.any_dtype> {sigArgTypes = ["Tensor", "Tensor", "Tensor?", "int[]", "int[]", "int[]", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
  return %3 : !numpy.ndarray<*:!numpy.any_dtype>
}
