// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file -verify-diagnostics

// The following test ensures that a tranposed convolution op is not
// lowered in the torch-to-tosa conversion pass.

func.func @forward(%input: !torch.vtensor<[1,64,1,100],f32>) -> !torch.vtensor<[1,64,2,200],f32> {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %weight = torch.vtensor.literal(dense<0.0> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
  %bias = torch.vtensor.literal(dense<0.0> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1x1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error@+1 {{failed to legalize operation 'torch.aten.convolution' that was explicitly marked illegal}}
  %output = torch.aten.convolution %input, %weight, %bias, %stride, %int1x1, %int1x1, %true, %int1x1, %int1 : !torch.vtensor<[1,64,1,100],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,2,200],f32>
  return %output : !torch.vtensor<[1,64,2,200],f32>
}

