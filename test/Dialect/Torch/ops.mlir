// RUN: npcomp-opt %s | npcomp-opt | FileCheck %s

func @kernel_call(%arg0 : si32, %arg1 : tensor<3x4xf32>) -> tensor<*xf32> {
  // CHECK: torch.kernel_call "somens::someop" %arg0, %arg1 : (si32, tensor<3x4xf32>) -> tensor<*xf32>
  %1 = torch.kernel_call "somens::someop" %arg0, %arg1 : (si32, tensor<3x4xf32>) -> (tensor<*xf32>) {
    sigArgTypes = [], sigRetTypes = [], sigIsVararg = false, sigIsVarret = false, sigIsMutable = false
  }
  return %1 : tensor<*xf32>
}


%bool_true = basicpy.bool_constant true
%num3_i64 = basicpy.numeric_constant 3 : i64
%num = basicpy.numeric_constant 4.250000e+01 : f64
%cst = constant dense<1.000000e+00> : tensor<1xf32>
%array = numpy.create_array_from_tensor %cst : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
func @f(%arg0: !torch.nn.Module) {
  return
}
%submodule = torch.nn_module {}

torch.nn_module {
  torch.attr "b", %bool_true : !basicpy.BoolType
  torch.attr "i", %num3_i64 : i64
  torch.attr "f", %num : f64
  torch.attr "t", %array : !numpy.ndarray<*:!numpy.any_dtype>
  torch.attr "submodule", %submodule : !torch.nn.Module
  torch.method "method", @f
}
