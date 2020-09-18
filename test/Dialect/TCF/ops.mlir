// RUN: npcomp-opt <%s | FileCheck %s --dump-input=fail

func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) {
  // CHECK: tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = tcf.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return
}

func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: tcf.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = tcf.matmul %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
