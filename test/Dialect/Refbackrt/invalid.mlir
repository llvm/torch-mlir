// RUN: npcomp-opt <%s -split-input-file -verify-diagnostics

refbackrt.module_metadata {
  // expected-error @+1 {{must reference a valid func}}
  refbackrt.func_metadata {funcName = @g, numInputs = 1 : i32, numOutputs = 0 : i32}
}


// -----

refbackrt.module_metadata {
  // expected-error @+1 {{must agree on number of inputs}}
  refbackrt.func_metadata {funcName = @f, numInputs = 1 : i32, numOutputs = 0 : i32}
}
func @f() { return }

// -----

refbackrt.module_metadata {
  // expected-error @+1 {{must agree on number of outputs}}
  refbackrt.func_metadata {funcName = @f, numInputs = 0 : i32, numOutputs = 1 : i32}
}
func @f() { return }
