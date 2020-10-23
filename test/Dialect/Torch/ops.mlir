// RUN: npcomp-opt %s | npcomp-opt | FileCheck %s

func @kernel_call(%arg0 : si32, %arg1 : tensor<3x4xf32>) -> tensor<*xf32> {
  // CHECK: %0 = torch.kernel_call "somens::someop" %arg0, %arg1 : (si32, tensor<3x4xf32>) -> tensor<*xf32>
  %1 = torch.kernel_call "somens::someop" %arg0, %arg1 : (si32, tensor<3x4xf32>) -> (tensor<*xf32>) {
    sigArgTypes = [], sigRetTypes = [], sigIsVararg = false, sigIsVarret = false, sigIsMutable = false
  }
  return %1 : tensor<*xf32>
}
