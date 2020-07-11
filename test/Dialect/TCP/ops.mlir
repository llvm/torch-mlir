// RUN: npcomp-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: tcp.global @foo dense<0.0{{.*}}> : tensor<10xf32>
tcp.global @foo dense<0.0> : tensor<10xf32>

func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: i32) {
  // CHECK: tcp.add
  %0 = "tcp.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = tcp.get_global_memref @foo : memref<10xf32>
  return
}
