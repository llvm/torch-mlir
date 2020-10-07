// RUN: npcomp-opt -lower-std-to-memref <%s -split-input-file | FileCheck %s --dump-input=fail

// If we also ran -lower-structural-to-memref, we could avoid all this casting
// stuff and make the output of the test cases cleaner, but we choose not to do
// that to make the test actually check what happens in practice.

// CHECK-LABEL:   func @extract_element
// CHECK:           %[[MEMREF:.*]] = refback.tensor_to_memref %arg0
// CHECK:           %[[RET:.*]] = load %[[MEMREF]][%arg1] : memref<?xf32>
// CHECK:           return %[[RET]] : f32
func @extract_element(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = extract_element %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}
// CHECK-LABEL:   func @tensor_from_elements(
// CHECK-SAME:                               %[[ARG0:.*]]: index,
// CHECK-SAME:                               %[[ARG1:.*]]: index) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = alloc()
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           store %[[ARG0]], %[[MEMREF]][%[[C0]]]
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           store %[[ARG1]], %[[MEMREF]][%[[C1]]]
// CHECK:           %[[RET:.*]] = refback.memref_to_tensor %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor_from_elements(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor_from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}


// CHECK-LABEL:   func @tensor_cast(
// CHECK-SAME:                      %[[ARG0:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = refback.tensor_to_memref %[[ARG0]] : tensor<?xindex> -> memref<?xindex>
// CHECK:           %[[CASTED:.*]] = memref_cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = refback.memref_to_tensor %[[CASTED]] : memref<2xindex> -> tensor<2xindex>
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor_cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor_cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor_load(
// CHECK-SAME:                      %[[ARG0:.*]]: memref<?xindex>) -> tensor<?xindex> {
// CHECK:           %[[RET:.*]] = refback.memref_to_tensor %[[ARG0]] : memref<?xindex> -> tensor<?xindex>
// CHECK:           return %[[RET]] : tensor<?xindex>
func @tensor_load(%arg0: memref<?xindex>) -> tensor<?xindex> {
  %0 = tensor_load %arg0 : memref<?xindex>
  return %0 : tensor<?xindex>
}
