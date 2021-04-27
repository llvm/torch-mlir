// RUN: npcomp-opt -torch-inline-global-slots -split-input-file %s | FileCheck %s

// CHECK-NOT: @readonly
torch.global_slot "private" @readonly : !numpy.ndarray<*:!numpy.any_dtype>  {
  %cst = constant dense<0.0> : tensor<1xf32>
  %0 = numpy.create_array_from_tensor %cst : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
  torch.global_slot.init %0 : !numpy.ndarray<*:!numpy.any_dtype>
}
// CHECK-LABEL: torch.global_slot @public
torch.global_slot @public : !numpy.ndarray<*:!numpy.any_dtype>  {
  %cst = constant dense<0.0> : tensor<2xf32>
  %0 = numpy.create_array_from_tensor %cst : (tensor<2xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
  torch.global_slot.init %0 : !numpy.ndarray<*:!numpy.any_dtype>
}
// CHECK-LABEL: torch.global_slot "private" @mutated
torch.global_slot "private" @mutated : !numpy.ndarray<*:!numpy.any_dtype>  {
  %cst = constant dense<0.0> : tensor<3xf32>
  %0 = numpy.create_array_from_tensor %cst : (tensor<3xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
  torch.global_slot.init %0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// CHECK-LABEL:   func @forward() -> (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>) {
func @forward() -> (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>) {
  // Inlined.
  // CHECK:           %[[CST:.*]] = constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK:           %[[ARRAY_CST:.*]] = numpy.create_array_from_tensor %[[CST]] : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
  %0 = torch.global_slot.get @readonly : !numpy.ndarray<*:!numpy.any_dtype>

  // Not inlined: potentially mutated by externals.
  // CHECK:           %[[PUBLIC:.*]] = torch.global_slot.get @public : !numpy.ndarray<*:!numpy.any_dtype>
  %1 = torch.global_slot.get @public : !numpy.ndarray<*:!numpy.any_dtype>

  // Not inlined: potentially mutated internally.
  // CHECK:           torch.global_slot.set @mutated = %[[ARRAY_CST]] : !numpy.ndarray<*:!numpy.any_dtype>
  // CHECK:           %[[MUTATED:.*]] = torch.global_slot.get @mutated : !numpy.ndarray<*:!numpy.any_dtype>
  torch.global_slot.set @mutated = %0 : !numpy.ndarray<*:!numpy.any_dtype>
  %2 = torch.global_slot.get @mutated : !numpy.ndarray<*:!numpy.any_dtype>

  // CHECK:           return %[[ARRAY_CST]], %[[PUBLIC]], %[[MUTATED]] : !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>
  return %0, %1, %2 : !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>
}
