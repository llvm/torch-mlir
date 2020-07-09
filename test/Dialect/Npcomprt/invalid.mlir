// RUN: npcomp-opt <%s -split-input-file -verify-diagnostics

npcomprt.module_metadata {
  // expected-error @+1 {{must reference a valid func}}
  npcomprt.func_metadata {funcName = @g, numInputs = 1 : i32, numOutputs = 0 : i32}
}


// -----

npcomprt.module_metadata {
  // expected-error @+1 {{must agree on number of inputs}}
  npcomprt.func_metadata {funcName = @f, numInputs = 1 : i32, numOutputs = 0 : i32}
}
func @f() { return }

// -----

npcomprt.module_metadata {
  // expected-error @+1 {{must agree on number of outputs}}
  npcomprt.func_metadata {funcName = @f, numInputs = 0 : i32, numOutputs = 1 : i32}
}
func @f() { return }
