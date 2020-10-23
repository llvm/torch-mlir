// RUN: npcomp-opt %s -aten-recognize-kernels -split-input-file |& FileCheck %s
// Note that this test is not exhaustive with respect to ops (since the facility
// is generic). Instead, it uses examplar ops to test various types of
// conversions.

// CHECK-LABEL: func @graph
func @graph(%arg0: !numpy.ndarray<*:?>, %arg1 : !numpy.ndarray<*:?>, %arg2 : si64) -> !numpy.ndarray<*:?> {
  // CHECK: %[[LHS:.*]] = numpy.copy_to_tensor %arg0
  // CHECK: %[[RHS:.*]] = numpy.copy_to_tensor %arg1
  // CHECK: %[[RESULT_IMM:.*]] = "aten.add"(%[[LHS]], %[[RHS]], %arg2) : (tensor<*x!basicpy.UnknownType>, tensor<*x!basicpy.UnknownType>, si64) -> tensor<*x!basicpy.UnknownType>
  // CHECK: %[[RESULT_MUT:.*]] = numpy.create_array_from_tensor %[[RESULT_IMM]] : (tensor<*x!basicpy.UnknownType>) -> !numpy.ndarray<*:?>
  %0 = torch.kernel_call "aten::add" %arg0, %arg1, %arg2 : (!numpy.ndarray<*:?>, !numpy.ndarray<*:?>, si64) -> !numpy.ndarray<*:?> {sigArgTypes = ["Tensor", "Tensor", "Scalar"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
  // CHECK: return %[[RESULT_MUT]]
  return %0 : !numpy.ndarray<*:?>
}
