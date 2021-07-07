// RUN: npcomp-opt -npcomp-verify-backend-contract -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

// CHECK: func @mm
func @mm(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {iree.module.export} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cst = constant 0.000000e+00 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %4 = cmpi eq, %1, %2 : index
  assert %4, "mismatching contracting dimension for aten.mm"
  %5 = linalg.init_tensor [%0, %3] : tensor<?x?xf32>
  %6 = linalg.fill(%cst, %5) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  %7 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %7 : tensor<?x?xf32>
}

// -----

// Basic check of error reporting.

// expected-error@+1 {{Module does not conform to npcomp's backend contract.}}
module {
  func @disallowed() {
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
// `tensor<?x!numpy.any_dtype>` is the problem here, which suggests
// that type inference didn't succeed, or insufficient type information
// was available.
//
// Ultimately, the output of this pass needs to be conveyed to the user
// in an understandable way, such as suggesting a particular place where
// a shape annotation is needed.

// expected-error@+1 {{Module does not conform to npcomp's backend contract.}}
module {
  func @disallowed(%arg0: tensor<?x!numpy.any_dtype>) -> tensor<?x!numpy.any_dtype> {
    // expected-error@+1 {{failed to legalize operation 'std.return'}}
    return %arg0 : tensor<?x!numpy.any_dtype>
  }
}
