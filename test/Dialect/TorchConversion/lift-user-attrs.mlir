// RUN: torch-mlir-opt %s --torch-lift-user-attrs | FileCheck %s

// Strip the `mlir.user.` prefix off any prefixed discardable attr on an op,
// re-attaching the value under the un-prefixed name.

// CHECK-LABEL: func.func @op_attrs
// CHECK:         arith.addf
// CHECK-SAME:    {domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64}
// CHECK-NOT:     mlir.user.
func.func @op_attrs(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b {mlir.user.domain_lower = -1.0 : f64, mlir.user.domain_upper = 1.0 : f64} : f32
  return %0 : f32
}

// -----

// Same lifting also happens on `func.func` per-arg attribute dictionaries.
// Stripped attr names must remain dialect-prefixed for the func verifier.

// CHECK-LABEL: func.func @arg_attrs
// CHECK-SAME:    %arg0: f32 {my.tag = "ciphertext", secret.secret}
// CHECK-NOT:     mlir.user.
func.func @arg_attrs(%arg0: f32 {mlir.user.secret.secret, mlir.user.my.tag = "ciphertext"}) -> f32 {
  return %arg0 : f32
}

// -----

// Discardable attrs without the `mlir.user.` prefix are left alone.

// CHECK-LABEL: func.func @other_attrs_untouched
// CHECK:         arith.addf
// CHECK-SAME:    {dialect.flag = 1 : i64}
// CHECK-NOT:     mlir.user.
func.func @other_attrs_untouched(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b {dialect.flag = 1 : i64} : f32
  return %0 : f32
}
