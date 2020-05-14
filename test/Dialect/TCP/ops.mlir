// RUN: npcomp-opt <%s | FileCheck %s --dump-input=fail

func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: i32) {
  // CHECK: tcp.add
  %0 = "tcp.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}
