// RUN: npcomp-opt -split-input-file %s -verify-diagnostics -allow-unregistered-dialect -numpy-public-functions-to-tensor | FileCheck --dump-input=fail %s

// CHECK-LABEL: legalConversion
module @legalConversion {
  // CHECK: @f(%arg0: tensor<3x?xf32>, %arg1: i32, %arg2: tensor<*xf32>) -> (i32, tensor<3x?xf32>, tensor<*xf32>)
  func @f(%arg0: !numpy.ndarray<[3,?]:f32>, %arg1: i32, %arg2: !numpy.ndarray<*:f32>) ->
      (i32, !numpy.ndarray<[3,?]:f32>, !numpy.ndarray<*:f32>) {
    // CHECK: %[[CREATE0:.+]] = numpy.create_array_from_tensor %arg0
    // CHECK: %[[CREATE1:.+]] = numpy.create_array_from_tensor %arg2
    // CHECK: %[[R0:.+]] = "unfoldable_arg0"(%[[CREATE0]]) : (!numpy.ndarray<[3,?]:f32>) -> !numpy.ndarray<[3,?]:f32>
    // CHECK: %[[R1:.+]] = "unfoldable_arg1"(%[[CREATE1]]) : (!numpy.ndarray<*:f32>) -> !numpy.ndarray<*:f32>
    %0 = "unfoldable_arg0"(%arg0) : (!numpy.ndarray<[3,?]:f32>) -> !numpy.ndarray<[3,?]:f32>
    %1 = "unfoldable_arg1"(%arg2) : (!numpy.ndarray<*:f32>) -> !numpy.ndarray<*:f32>
    // CHECK: %[[COPY0:.+]] = numpy.copy_to_tensor %[[R0]]
    // CHECK: %[[COPY1:.+]] = numpy.copy_to_tensor %[[R1]]
    // CHECK: return %arg1, %[[COPY0]], %[[COPY1]] : i32, tensor<3x?xf32>, tensor<*xf32>
    return %arg1, %0, %1 : i32, !numpy.ndarray<[3,?]:f32>, !numpy.ndarray<*:f32>
  }
}

// -----
// CHECK-LABEL: @nonPublic
module @nonPublic {
  // CHECK: @f(%arg0: !numpy.ndarray<[3,?]:f32>) -> !numpy.ndarray<[3,?]:f32>
  func @f(%arg0: !numpy.ndarray<[3,?]:f32>) -> (!numpy.ndarray<[3,?]:f32>)
      attributes { sym_visibility = "private" } {
    return %arg0 : !numpy.ndarray<[3,?]:f32>
  }
}

// -----
// CHECK-LABEL: @called
module @called {
  // CHECK: @f(%arg0: !numpy.ndarray<*:f32>) -> !numpy.ndarray<*:f32>
  // expected-warning @+1 {{unimplemented: cannot convert}}
  func @f(%arg0: !numpy.ndarray<*:f32>) -> !numpy.ndarray<*:f32> {
    return %arg0 : !numpy.ndarray<*:f32>
  }

  func @caller(%arg0: !numpy.ndarray<*:f32>) -> !numpy.ndarray<*:f32>
    attributes { sym_visibility = "private" } {
    %0 = call @f(%arg0) : (!numpy.ndarray<*:f32>) -> !numpy.ndarray<*:f32>
    return %0 : !numpy.ndarray<*:f32>
  }
}
