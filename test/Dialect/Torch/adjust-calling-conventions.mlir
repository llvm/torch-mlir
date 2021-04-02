// RUN: npcomp-opt -torch-adjust-calling-conventions -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !numpy.ndarray<[2,3,?]:f32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
// CHECK:           %[[RET:.*]] = numpy.static_info_cast %[[ARG]] : !numpy.ndarray<[2,3,?]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           return %[[RET]] : !numpy.ndarray<*:!numpy.any_dtype>
func @basic(%arg0: !numpy.ndarray<*:!numpy.any_dtype> {torch.type_bound = !numpy.ndarray<[2,3,?]:f32>}) -> !numpy.ndarray<*:!numpy.any_dtype> {
  return %arg0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// CHECK-LABEL:   func @no_type_bound(
// CHECK-SAME:                        %[[ARG:.*]]: !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype> {
// CHECK:           return %[[ARG]] : !numpy.ndarray<*:!numpy.any_dtype>
func @no_type_bound(%arg0: !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  return %arg0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// CHECK-LABEL:   func @call(
// CHECK-SAME:               %[[ARG:.*]]: !numpy.ndarray<[2,3,?]:f32>) -> !numpy.ndarray<*:!numpy.any_dtype> {
// CHECK:           %[[SHAPE_ERASED:.*]] = numpy.static_info_cast %[[ARG]] : !numpy.ndarray<[2,3,?]:f32> to !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           %[[SHAPED:.*]] = numpy.static_info_cast %[[SHAPE_ERASED]] : !numpy.ndarray<*:!numpy.any_dtype> to !numpy.ndarray<[2,3,?]:f32>
// CHECK:           %[[RES:.*]] = call @call(%[[SHAPED]]) : (!numpy.ndarray<[2,3,?]:f32>) -> !numpy.ndarray<*:!numpy.any_dtype>
// CHECK:           return %[[SHAPE_ERASED]] : !numpy.ndarray<*:!numpy.any_dtype>
func @call(%arg0: !numpy.ndarray<*:!numpy.any_dtype> {torch.type_bound = !numpy.ndarray<[2,3,?]:f32>}) -> !numpy.ndarray<*:!numpy.any_dtype> {
  %0 = call @call(%arg0) : (!numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype>
  return %arg0 : !numpy.ndarray<*:!numpy.any_dtype>
}

// CHECK-LABEL:   func @none_return() {
// CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
// CHECK:           return
func @none_return() -> !basicpy.NoneType {
  %1 = basicpy.singleton : !basicpy.NoneType
  return %1 : !basicpy.NoneType
}

// CHECK-LABEL:   func @none_call_return() {
// CHECK:           call @none_return() : () -> ()
// CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
// CHECK:           "test.use"(%[[NONE]]) : (!basicpy.NoneType) -> ()
// CHECK:           return
func @none_call_return() {
  %0 = call @none_return() : () -> !basicpy.NoneType
  "test.use"(%0) : (!basicpy.NoneType) -> ()
  return
}
