// RUN: torch-mlir-opt -split-input-file -verify-diagnostics %s -torch-verify-conversion-to-value-semantics

// -----

func.func @result_is_non_value_tensor(%arg: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  // @expected-error@+1 {{found a non-value tensor type, this is likely due to a missing case in the MaximizeValueSemantics pass}}
  %neg = torch.aten.neg %arg : !torch.vtensor<[2],f32> -> !torch.tensor
  return %arg : !torch.vtensor<[2],f32>
}
