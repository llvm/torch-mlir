// RUN: npcomp-opt -split-input-file -verify-diagnostics %s

func @numeric_constant_string_attr() {
  // expected-error @+1 {{op requires 'value' to be an integer constant}}
  %0 = "basicpy.numeric_constant"() {value="somestring" : i32} : () -> (i32)
  return
}

// -----
func @numeric_constant_bool() {
  // expected-error @+1 {{cannot have an i1 type}}
  %0 = "basicpy.numeric_constant"() {value = true} : () -> (i1)
  return
}

// -----
func @numeric_constant_mismatch_int() {
  // expected-error @+1 {{op requires 'value' to be a floating point constant}}
  %0 = "basicpy.numeric_constant"() {value = 1 : i32} : () -> (f64)
  return
}

// -----
func @numeric_constant_mismatch_float() {
  // expected-error @+1 {{op requires 'value' to be an integer constant}}
  %0 = "basicpy.numeric_constant"() {value = 1.0 : f32} : () -> (i32)
  return
}

// -----
func @numeric_constant_complex_wrong_arity() {
  // expected-error @+1 {{op requires 'value' to be a two element array of floating point complex number components}}
  %3 = basicpy.numeric_constant [2.0 : f32] : complex<f32>
  return
}

// -----
func @numeric_constant_complex_mismatch_type_real() {
  // expected-error @+1 {{op requires 'value' to be a two element array of floating point complex number components}}
  %3 = basicpy.numeric_constant [2.0 : f64, 3.0 : f32] : complex<f32>
  return
}

// -----
func @numeric_constant_complex_mismatch_type_imag() {
  // expected-error @+1 {{op requires 'value' to be a two element array of floating point complex number components}}
  %3 = basicpy.numeric_constant [2.0 : f32, 3.0 : f16] : complex<f32>
  return
}
