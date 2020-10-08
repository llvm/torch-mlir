// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s

// CHECK-LABEL: refback.global @foo dense<0.0{{.*}}> : tensor<10xf32>
refback.global @foo dense<0.0> : tensor<10xf32>

// CHECK-LABEL: func @global
func @global() {
  // CHECK: refback.get_global_memref @foo : memref<10xf32>
  %0 = refback.get_global_memref @foo : memref<10xf32>
  return
}

// CHECK-LABEL:      func @shaped_results
// CHECK-NEXT:   %[[RET:.*]] = refback.shaped_results %arg1 {
// CHECK-NEXT:     %[[VAL:.*]] =
// CHECK-NEXT:     refback.yield %[[VAL]] : tensor<?x?xf32>
// CHECK-NEXT:   } : tensor<?xindex> -> tensor<?x?xf32>
// CHECK-NEXT:   return %[[RET]] : tensor<?x?xf32>
// CHECK-NEXT: }
func @shaped_results(%arg0: tensor<?x?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %add = refback.shaped_results %arg1 {
    %0 = tcp.add %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    refback.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}
