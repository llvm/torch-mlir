// RUN: torch-mlir-opt -torch-verify-tosa-backend-contract -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

// CHECK: func.func @tanh
func.func @tanh(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tosa.tanh %arg0 : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// Basic check of error reporting.

// expected-error@+1 {{Module does not conform to the TOSA backend contract.}}
module {
  func.func @disallowed() {
    // expected-error@+1 {{failed to legalize operation 'unknown_dialect.unknown_op'}}
    "unknown_dialect.unknown_op"() : () -> ()
    return
  }
}

// -----

// TODO: Improve these errors to give more exact reporting.
//
// The reporting we inherit from dialect conversion is not precise.
// For example, here we want it to explicitly call out that
// `!torch.tensor` is the problem here, which suggests
// that type inference didn't succeed, or insufficient type information
// was available.
//
// Ultimately, the output of this pass needs to be conveyed to the user
// in an understandable way, such as suggesting a particular place where
// a shape annotation is needed.

// expected-error@+1 {{Module does not conform to the TOSA backend contract.}}
module {
  func.func @disallowed(%arg0: !torch.tensor) -> !torch.tensor {
    // expected-error@+1 {{failed to legalize operation 'func.return'}}
    return %arg0 : !torch.tensor
  }
}
