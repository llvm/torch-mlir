// RUN: torch-mlir-opt -torch-verify-backend-contract-no-decompositions -split-input-file -verify-diagnostics %s

func.func @f(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
  // expected-error @below {{unsupported by backend contract: tensor with unknown rank}}
  // expected-note @below {{this is likely due to a missing transfer function}}
  %t = torch.aten.t %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor
  return %t : !torch.vtensor
}

// -----

// expected-error @below {{invalid dtype 'i9'}}
func.func @bad_element_type(%arg: !torch.vtensor<[?],i9>) -> !torch.vtensor<[?],i9> {
  return %arg : !torch.vtensor<[?],i9>
}

// -----

// expected-error @below {{unsupported by backend contract: non-value tensor type}}
// expected-note @below {{this is likely due to a missing case in the MaximizeValueSemantics pass}}
func.func @non_value_tensor(%arg0: !torch.tensor) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// -----

func.func @valid_tuple(%arg0: !torch.vtensor<[?],f32>) -> !torch.tuple<vtensor<[?],f32>> {
  %0 = torch.prim.TupleConstruct %arg0 : !torch.vtensor<[?],f32> -> !torch.tuple<vtensor<[?],f32>>
  return %0 : !torch.tuple<vtensor<[?],f32>>
}

// -----

func.func @valid_multiple_ret_values(%arg0: !torch.vtensor<[?],f32>) -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>) {
  return %arg0, %arg0 : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>
}
