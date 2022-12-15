// RUN: torch-mlir-opt -torch-verify-backend-contract -split-input-file -verify-diagnostics %s
func.func @f(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  // expected-error @+2 {{found an op that was marked as backend illegal}}
  // expected-note @+1 {{this is likely due to}}
  %t = torch.aten.t %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %t : !torch.vtensor<[?,?],f32>
}
