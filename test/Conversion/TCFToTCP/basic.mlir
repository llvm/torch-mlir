// RUN: npcomp-opt <%s -convert-tcf-to-tcp | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @f
func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // Just the lightest sanity check.
  // CHECK: tcp.add
  %0 = "tcf.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
