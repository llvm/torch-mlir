// RUN: npcomp-opt <%s -pass-pipeline=e2e-lowering-pipeline | FileCheck %s --dump-input=fail
// RUN: npcomp-opt <%s -pass-pipeline=e2e-lowering-pipeline{optimize} | FileCheck %s --dump-input=fail

// -----
// CHECK-LABEL: func @global_add
func @global_add() -> tensor<2xf32> {
  %c34 = constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
  %c12 = constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %0 = tcf.add %c34, %c12 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %1 = tcf.add %c12, %0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}
