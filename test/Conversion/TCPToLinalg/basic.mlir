// RUN: npcomp-opt <%s -convert-tcp-to-linalg | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic
  %0 = "tcp.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}