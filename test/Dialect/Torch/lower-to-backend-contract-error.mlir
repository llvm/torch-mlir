// RUN: torch-mlir-opt -torch-lower-to-backend-contract -split-input-file -verify-diagnostics %s

torch.global_slot.module_initializer {
  %0 = torch.constant.int 1
  // expected-error @+2 {{unsupported by backend contract: module initializers}}
  // expected-note @+1 {{this is likely due to}}
  torch.initialize.global_slots [
    @slot0(%0 : !torch.int)
  ]
}
torch.global_slot @slot0 : !torch.int


// -----

// expected-error @+2 {{unsupported by backend contract: non-value tensor type}}
// expected-note @+1 {{this is likely due to}}
func.func @f(%arg0: !torch.tensor) {
  return
}

// -----

// expected-error @+2 {{unsupported by backend contract: tensor with unknown rank}}
// expected-note @+1 {{this is likely due to}}
func.func @f(%arg0: !torch.vtensor<*,f32>) {
  return
}

// -----

// expected-error @+2 {{unsupported by backend contract: tensor with unknown dtype}}
// expected-note @+1 {{this is likely due to}}
func.func @f(%arg0: !torch.vtensor<[],unk>) {
  return
}

// -----

// expected-error @+1 {{unsupported by backend contract: type '!torch.any'}}
func.func @f(%arg0: !torch.any) {
  return
}

// -----

// Decomposition of `aten.t` fails if `inputRank > 2`
func.func @f(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  // expected-error @+2 {{found an op that was marked as backend illegal}}
  // expected-note @+1 {{this is likely due to}}
  %t = torch.aten.t %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %t : !torch.vtensor<[?,?,?],f32>
}

// -----

// Test case: checking of op results.
// TODO: In theory we could diagnose every single value, but for now we bail out on the first one.

func.func @f(%arg0: !torch.bool, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[7],f32>) -> !torch.vtensor<*,f32> {
  // expected-error @+2 {{unsupported by backend contract: tensor with unknown rank}}
  // expected-note @+1 {{this is likely due to}}
  %0 = torch.prim.If %arg0 -> (!torch.vtensor<*,f32>) {
    %1 = torch.tensor_static_info_cast %arg1 : !torch.vtensor<[],f32> to !torch.vtensor<*,f32>
    torch.prim.If.yield %1 : !torch.vtensor<*,f32>
  } else {
    %2 = torch.tensor_static_info_cast %arg2 : !torch.vtensor<[7],f32> to !torch.vtensor<*,f32>
    torch.prim.If.yield %2 : !torch.vtensor<*,f32>
  }
  return %0 : !torch.vtensor<*,f32>
}
