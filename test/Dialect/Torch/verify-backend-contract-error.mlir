// RUN: torch-mlir-opt -torch-verify-backend-contract-no-decompositions -split-input-file -verify-diagnostics %s
func.func @f(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
  // expected-error @below {{unsupported by backend contract: tensor with unknown rank}}
  // expected-note @below {{this is likely due to a missing transfer function}}
  %t = torch.aten.t %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor
  return %t : !torch.vtensor
}
