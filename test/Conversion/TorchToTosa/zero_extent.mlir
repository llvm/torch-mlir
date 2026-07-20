// RUN: torch-mlir-opt -convert-torch-to-tosa -split-input-file -verify-diagnostics %s

// expected-error @below {{TOSA conversion does not support zero-extent tensor type '!torch.vtensor<[0,1],f32>'}}
func.func @zero_extent_function_signature(%arg0: !torch.vtensor<[0,1],f32>) -> !torch.vtensor<[0,1],f32> {
  return %arg0 : !torch.vtensor<[0,1],f32>
}
