// RUN: npcomp-opt -lower-alloc-memref-ops <%s | FileCheck %s

// CHECK-LABEL: func @basic
func @basic(%arg0: !shape.shape) {
  // CHECK: %[[E:.*]] = tcp.get_extent %arg0, 0
  // CHECK: alloc(%[[E]])
  %0 = tcp.alloc_memref %arg0 : memref<?xf32>
  return
}

// CHECK: func @all_static(%arg0: !shape.shape)
func @all_static(%arg0: !shape.shape) {
  // CHECK-NOT: tcp.get_extent
  // CHECK: alloc()
  %0 = tcp.alloc_memref %arg0 : memref<3x4x5xf32>
  return
}

// CHECK: func @some_static(%arg0: !shape.shape)
func @some_static(%arg0: !shape.shape) {
  // CHECK: %[[E1:.*]] = tcp.get_extent %arg0, 1
  // CHECK: %[[E3:.*]] = tcp.get_extent %arg0, 3
  // CHECK: alloc(%[[E1]], %[[E3]])
  %0 = tcp.alloc_memref %arg0 : memref<3x?x5x?x7xf32>
  return
}
