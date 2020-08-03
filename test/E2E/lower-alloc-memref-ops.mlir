// RUN: npcomp-opt -split-input-file -lower-alloc-memref-ops <%s | FileCheck %s

// CHECK-LABEL: func @basic
func @basic(%arg0: tensor<?xindex>) -> memref<?xf32> {
  // CHECK: %[[I:.*]] = constant 0 : index
  // CHECK: %[[E:.*]] = shape.get_extent %arg0, %[[I]]
  // CHECK: alloc(%[[E]])
  %0 = tcp.alloc_memref %arg0 : memref<?xf32>
  return %0 : memref<?xf32>
}

// -----
// CHECK-LABEL: func @all_static
func @all_static(%arg0: tensor<?xindex>) -> memref<3x4x5xf32> {
  // CHECK-NOT: shape.get_extent
  // CHECK: alloc()
  %0 = tcp.alloc_memref %arg0 : memref<3x4x5xf32>
  return %0 : memref<3x4x5xf32>
}

// -----
// CHECK-LABEL: func @some_static
func @some_static(%arg0: tensor<?xindex>) -> memref<3x?x5x?x7xf32> {
  // CHECK-DAG: %[[I1:.*]] = constant 1 : index
  // CHECK-DAG: %[[E1:.*]] = shape.get_extent %arg0, %[[I1]]
  // CHECK-DAG: %[[I3:.*]] = constant 3 : index
  // CHECK-DAG: %[[E3:.*]] = shape.get_extent %arg0, %[[I3]]
  // CHECK: alloc(%[[E1]], %[[E3]])
  %0 = tcp.alloc_memref %arg0 : memref<3x?x5x?x7xf32>
  return %0 : memref<3x?x5x?x7xf32>
}
