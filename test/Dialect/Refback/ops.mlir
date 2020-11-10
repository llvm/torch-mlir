// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s

// CHECK-LABEL: @alloc_memref
func @alloc_memref(%arg0: tensor<?xindex>) {
  // CHECK: refback.alloc_memref
  %0 = refback.alloc_memref %arg0 : memref<?xf32>
  return
}
