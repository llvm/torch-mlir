// RUN: torch-mlir-opt -torch-verify-backend-contract-no-decompositions -split-input-file -verify-diagnostics %s
func.func @forward(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor {
  %none = torch.constant.none
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[3,5],f32> to !torch.vtensor<*,f32>
  %1 = torch.copy.to_tensor %0 : !torch.tensor<*,f32>
  // expected-error @+1 {{unsupported by backend contract: Unimplemented operator 'an.unimplemented.op'}}
  %2 = torch.operator "an.unimplemented.op"(%1, %1, %none) : (!torch.tensor<*,f32>, !torch.tensor<*,f32>, !torch.none) -> !torch.tensor
  %3 = torch.copy.to_vtensor %2 : !torch.vtensor
  return %3 : !torch.vtensor
}
