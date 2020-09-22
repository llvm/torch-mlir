// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: tcp.global @foo dense<0.0{{.*}}> : tensor<10xf32>
tcp.global @foo dense<0.0> : tensor<10xf32>

// CHECK-LABEL: func @global
func @global() {
  // CHECK: tcp.get_global_memref @foo : memref<10xf32>
  %0 = tcp.get_global_memref @foo : memref<10xf32>
  return
}

// CHECK-LABEL: func @binary_elementwise
func @binary_elementwise(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: i32) {
  // CHECK: tcp.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tcp.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = tcp.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = tcp.max %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}

// CHECK-LABEL: func @matmul
func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: tcp.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = tcp.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:      func @shaped_results
// CHECK-NEXT:   %[[RET:.*]] = tcp.shaped_results %arg1 {
// CHECK-NEXT:     %[[VAL:.*]] =
// CHECK-NEXT:     tcp.yield %[[VAL]] : tensor<?x?xf32>
// CHECK-NEXT:   } : tensor<?xindex> -> tensor<?x?xf32>
// CHECK-NEXT:   return %[[RET]] : tensor<?x?xf32>
// CHECK-NEXT: }
func @shaped_results(%arg0: tensor<?x?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %add = tcp.shaped_results %arg1 {
    %0 = tcp.add %arg0, %arg0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    tcp.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}
