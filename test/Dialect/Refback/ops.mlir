// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s

// CHECK-LABEL: refback.global @foo dense<0.0{{.*}}> : tensor<10xf32>
refback.global @foo dense<0.0> : tensor<10xf32>

// CHECK-LABEL: func @global
func @global() {
  // CHECK: refback.get_global_memref @foo : memref<10xf32>
  %0 = refback.get_global_memref @foo : memref<10xf32>
  return
}
