// RUN: npcomp-opt <%s -pass-pipeline=e2e-lowering-pipeline | FileCheck %s --dump-input=fail
// RUN: npcomp-opt <%s -pass-pipeline=e2e-lowering-pipeline{optimize} | FileCheck %s --dump-input=fail

// This is the simplest case, which is easy to stare at for debugging
// purposes.

// CHECK-LABEL: func @rank1
func @rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tcf.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: func @multiple_ops
func @multiple_ops(%arg0: tensor<f32>, %arg1: tensor<?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tcf.add"(%arg1, %arg2) : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tcf.add"(%arg0, %0) : (tensor<f32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
