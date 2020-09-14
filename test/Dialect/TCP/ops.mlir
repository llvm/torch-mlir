// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: tcp.global @foo dense<0.0{{.*}}> : tensor<10xf32>
tcp.global @foo dense<0.0> : tensor<10xf32>

func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: i32) {
  // CHECK: tcp.add
  %0 = "tcp.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = tcp.get_global_memref @foo : memref<10xf32>
  return
}

// CHECK-LABEL:      func @g
// CHECK-NEXT:   %[[RET:.*]] = tcp.shaped_results %arg1 {
// CHECK-NEXT:     %[[VAL:.*]] =
// CHECK-NEXT:     tcp.yield %[[VAL]] : tensor<?x?xf32>
// CHECK-NEXT:   } : tensor<?xindex> -> tensor<?x?xf32>
// CHECK-NEXT:   return %[[RET]] : tensor<?x?xf32>
// CHECK-NEXT: }
func @g(%arg0: tensor<?x?xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  %add = tcp.shaped_results %arg1 {
    %0 = "tcp.add"(%arg0, %arg0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    tcp.yield %0 : tensor<?x?xf32>
  } : tensor<?xindex> -> tensor<?x?xf32>
  return %add : tensor<?x?xf32>
}
