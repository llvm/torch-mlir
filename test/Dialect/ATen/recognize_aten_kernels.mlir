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


// -----
// CHECK-LABEL: func @nll_loss2d_forward
// Contains a Tensor? type mapped to None.
func @nll_loss2d_forward(
    %arg0: !numpy.ndarray<[3,4,8,8]:f32>,
    %arg1: !numpy.ndarray<[3,8,8]:i64>,
    %arg2: !basicpy.NoneType,
    %arg3: i64,
    %arg4: i64) -> (!numpy.ndarray<[]:f32>, !numpy.ndarray<[]:f32>) {
  // CHECK: %[[TARG0:.*]] = numpy.copy_to_tensor %arg0
  // CHECK: %[[TARG1:.*]] = numpy.copy_to_tensor %arg1
  // CHECK: %[[TOUTPUT:.*]], %[[TTOTAL_WEIGHT:.*]] = "aten.nll_loss2d_forward"(%[[TARG0]], %[[TARG1]], %arg2, %arg3, %arg4) : (tensor<3x4x8x8xf32>, tensor<3x8x8xi64>, !basicpy.NoneType, i64, i64) -> (tensor<f32>, tensor<f32>)
  // CHECK: %[[AOUTPUT:.*]] = numpy.create_array_from_tensor %[[TOUTPUT]]
  // CHECK: %[[ATOTAL_WEIGHT:.*]] =  numpy.create_array_from_tensor %[[TTOTAL_WEIGHT]]
  %0:2 = torch.kernel_call "aten::nll_loss2d_forward"
      %arg0, %arg1, %arg2, %arg3, %arg4 :
      (!numpy.ndarray<[3,4,8,8]:f32>, !numpy.ndarray<[3,8,8]:i64>, !basicpy.NoneType, i64, i64) ->
      (!numpy.ndarray<[]:f32>, !numpy.ndarray<[]:f32>)
      {sigArgTypes = ["Tensor", "Tensor", "Tensor?", "int", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor", "Tensor"]}
  // CHECK: return %[[AOUTPUT]], %[[ATOTAL_WEIGHT]]
  return %0#0, %0#1 : !numpy.ndarray<[]:f32>, !numpy.ndarray<[]:f32>
}

// -----
// CHECK-LABEL: func @convolution
// Contains a Tensor?, bool, int and list types.
func @convolution(
    %arg0: !numpy.ndarray<[3,16,10,10]:f32>, %arg1: !numpy.ndarray<[4,16,3,3]:f32>,
    %arg2: !numpy.ndarray<[4]:f32>, %arg3: !basicpy.ListType, %arg4: !basicpy.ListType,
    %arg5: !basicpy.ListType, %arg6: i1, %arg7: !basicpy.ListType, %arg8: i64) -> !numpy.ndarray<[3,4,8,8]:f32> {
  // CHECK: %[[TARG0:.*]] = numpy.copy_to_tensor %arg0
  // CHECK: %[[TARG1:.*]] = numpy.copy_to_tensor %arg1
  // CHECK: %[[TARG2:.*]] = numpy.copy_to_tensor %arg2
  // CHECK: %[[TRESULT:.*]] = "aten.convolution"(%[[TARG0]], %[[TARG1]], %[[TARG2]], %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (tensor<3x16x10x10xf32>, tensor<4x16x3x3xf32>, tensor<4xf32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i1, !basicpy.ListType, i64) -> tensor<3x4x8x8xf32>
  %0 = torch.kernel_call "aten::convolution"
      %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 :
      (!numpy.ndarray<[3,16,10,10]:f32>, !numpy.ndarray<[4,16,3,3]:f32>,
       !numpy.ndarray<[4]:f32>, !basicpy.ListType, !basicpy.ListType,
       !basicpy.ListType, i1, !basicpy.ListType, i64) -> !numpy.ndarray<[3,4,8,8]:f32>
       {sigArgTypes = ["Tensor", "Tensor", "Tensor?", "int[]", "int[]", "int[]", "bool", "int[]", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
  return %0 : !numpy.ndarray<[3,4,8,8]:f32>
}

// -----
// CHECK-LABEL: func @convolution_backward
// Interesting because it has optional tensor returns.
func @convolution_backward(
    %arg0: !numpy.ndarray<[3,4,8,8]:f32>,
    %arg1: !numpy.ndarray<[3,16,10,10]:f32>,
    %arg2: !numpy.ndarray<[4,16,3,3]:f32>,
    %arg3: !basicpy.ListType,
    %arg4: !basicpy.ListType,
    %arg5: !basicpy.ListType,
    %arg6: i1,
    %arg7: !basicpy.ListType,
    %arg8: i64,
    %arg9: !basicpy.ListType) -> (!basicpy.NoneType, !numpy.ndarray<[4,16,3,3]:f32>, !numpy.ndarray<[4]:f32>) {
  // CHECK: %[[GRAD_INPUT:.*]], %[[GRAD_WEIGHT:.*]], %[[GRAD_BIAS:.*]] = "aten.convolution_backward"
  // Note that this kernel call masks out the input gradients, which will return as NoneType.
  %0:3 = torch.kernel_call "aten::convolution_backward"
    %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 :
    (!numpy.ndarray<[3,4,8,8]:f32>, !numpy.ndarray<[3,16,10,10]:f32>, !numpy.ndarray<[4,16,3,3]:f32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i1, !basicpy.ListType, i64, !basicpy.ListType) ->
    (!basicpy.NoneType, !numpy.ndarray<[4,16,3,3]:f32>, !numpy.ndarray<[4]:f32>) {sigArgTypes = ["Tensor", "Tensor", "Tensor", "int[]", "int[]", "int[]", "bool", "int[]", "int", "bool[]"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor", "Tensor", "Tensor"]}
  // CHECK: %[[AGRAD_WEIGHT:.*]] = numpy.create_array_from_tensor %[[GRAD_WEIGHT]]
  // CHECK: %[[AGRAD_BIAS:.*]] = numpy.create_array_from_tensor %[[GRAD_BIAS]]
  // Key thing: The return returns the raw NoneType from the masked input gradient
  // and it does not get converted to an array.
  // CHECK: return %[[GRAD_INPUT]], %[[AGRAD_WEIGHT]], %[[AGRAD_BIAS]]
  return %0#0, %0#1, %0#2 : !basicpy.NoneType, !numpy.ndarray<[4,16,3,3]:f32>, !numpy.ndarray<[4]:f32>
}

// -----
// CHECK-LABEL: func @conv2d
// Contains a Tensor, Tensor, Tensor?, int[], int[] int[], int
func @conv2d(%arg0: !numpy.ndarray<*:!numpy.any_dtype>, %arg1: !numpy.ndarray<*:!numpy.any_dtype>, %arg2: !numpy.ndarray<*:!numpy.any_dtype>, %arg3: !basicpy.ListType, %arg4: !basicpy.ListType, %arg5: !basicpy.ListType, %arg6: i64) -> !numpy.ndarray<*:!numpy.any_dtype> {
  // CHECK: %[[TARG0:.*]] = numpy.copy_to_tensor %arg0
  // CHECK: %[[TARG1:.*]] = numpy.copy_to_tensor %arg1
  // CHECK: %[[TARG2:.*]] = numpy.copy_to_tensor %arg2
  // CHECK: %[[TRESULT:.*]] = "aten.conv2d"(%[[TARG0]], %[[TARG1]], %[[TARG2]], %arg3, %arg4, %arg5, %arg6) : (tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>, tensor<*x!numpy.any_dtype>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> tensor<*x!numpy.any_dtype>
  %0 = torch.kernel_call "aten::conv2d" %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6: (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> !numpy.ndarray<*:!numpy.any_dtype> {sigArgTypes = ["Tensor", "Tensor", "Tensor?", "int[]", "int[]", "int[]", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
  return %0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// -----
// CHECK-LABEL: func @copy_inplace
// Mutable/in-place op conversion, dropping result.
func @copy_inplace(%arg0: !numpy.ndarray<[4]:f32>, %arg1: !numpy.ndarray<[4]:f32>) -> !numpy.ndarray<[4]:f32> {
  // CHECK: %[[TARG1:.*]] = numpy.copy_to_tensor %arg1
  // CHECK: "aten.copy.inplace"(%arg0, %[[TARG1]]) : (!numpy.ndarray<[4]:f32>, tensor<4xf32>) -> ()
  %0 = torch.kernel_call "aten::copy_" %arg0, %arg1 : (!numpy.ndarray<[4]:f32>, !numpy.ndarray<[4]:f32>) -> !numpy.ndarray<[4]:f32> {sigArgTypes = ["Tensor", "Tensor", "bool"], sigIsMutable = true, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
  // CHECK: return %arg0
  return %0 : !numpy.ndarray<[4]:f32>
}
