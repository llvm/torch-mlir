// RUN: npcomp-opt -lower-to-memref-abi <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @identity
func @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: return %arg0 : memref<*xf32>
  return %arg0 : tensor<?xf32>
}

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[VAL_1:.*]]: memref<*xf32>) -> memref<*xf32> {
func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {

  // CHECK: %[[VAL_2:.*]] = memref_cast %[[VAL_1]] : memref<*xf32> to memref<?xf32>
  // CHECK: %[[VAL_3:.*]] = dim %[[VAL_2]], 0 : memref<?xf32>
  // CHECK: %[[VAL_4:.*]] = shape.from_extents %[[VAL_3]]
  %shape = shape.shape_of %arg0 : tensor<?xf32>

  // CHECK: %[[VAL_5:.*]] = tcp.alloc_memref %[[VAL_4]] : memref<?xf32>
  %memref = tcp.alloc_memref %shape : memref<?xf32>

  // CHECK: %[[VAL_6:.*]] = memref_cast %[[VAL_1]] : memref<*xf32> to memref<?xf32>
  // CHECK: linalg.copy(%[[VAL_6]], %[[VAL_5]]) : memref<?xf32>, memref<?xf32>
  tensor_store %arg0, %memref : memref<?xf32>

  // CHECK: %[[VAL_7:.*]] = memref_cast %[[VAL_5]] : memref<?xf32> to memref<*xf32>
  %ret = tensor_load %memref : memref<?xf32>

  // CHECK: return %[[VAL_7]] : memref<*xf32>
  return %ret: tensor<?xf32>
}
