// RUN: torch-mlir-opt -torch-verify-stablehlo-backend-contract -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s
// REQUIRES: stablehlo

// CHECK: func.func @tanh
func.func @tanh(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.tanh %arg0 : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Basic check of error reporting.

// expected-error@+1 {{Module does not conform to the Stablehlo backend contract.}}
module {
  func.func @disallowed() {
    // expected-error@+1 {{failed to legalize operation 'unknown_dialect.unknown_op'}}
    "unknown_dialect.unknown_op"() : () -> ()
    return
  }
}

// -----

// A non-shaped function argument (e.g. `!torch.optional`) that is dead --
// referenced by no op in the body -- should be rejected.

// expected-error@+1 {{Module does not conform to the Stablehlo backend contract.}}
module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @dead_optional_arg(%arg0: !torch.optional<vtensor<[4],f32>>) -> tensor<1xi64> {
    %cst = arith.constant dense<1> : tensor<1xi64>
    return %cst : tensor<1xi64>
  }
}
