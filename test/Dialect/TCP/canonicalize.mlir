// RUN: npcomp-opt -canonicalize <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @tensor_to_memref
func @tensor_to_memref_fold(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NEXT: return %arg0 : memref<?xf32>
  %0 = tcp.memref_to_tensor %arg0 : memref<?xf32> -> tensor<?xf32>
  %1 = tcp.tensor_to_memref %0 : tensor<?xf32> -> memref<?xf32>
  return %1 : memref<?xf32>
}
