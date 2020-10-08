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

// -----

refbackrt.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{must reference a valid refbackrt.global}}
    refbackrt.get_global @nonexistent_symbol : memref<*xf32>
    return
}

// -----

refbackrt.global @g dense<0.0> : tensor<2xf32>

func @f() {
    // expected-error @+1 {{inconsistent with element type of global}}
    refbackrt.get_global @g : memref<*xi8>
    return
}
