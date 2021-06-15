// RUN: npcomp-opt <%s -split-input-file -verify-diagnostics

// -----

torch.class_type @c {}
%0 = torch.nn_module {
  // expected-error @+1 {{'func' op is not allowed inside 'torch.nn_module'}}
  func @f()
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {}
%c0 = torch.constant.int 0 : i64
// expected-error @+1 {{number of 'torch.slot's in a 'torch.nn_module' must match number of 'torch.attr's in the corresponding 'torch.class_type'}}
%0 = torch.nn_module {
  torch.slot "f", %c0 : i64
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {
  // expected-note @+1 {{see torch.attr at corresponding index 0 here}}
  torch.attr "g" : i64
}
%c0 = torch.constant.int 0 : i64
%0 = torch.nn_module {
  // expected-error @+1 {{'torch.slot' op is expected to match type and name of 'torch.attr "g" : i64'}}
  torch.slot "f", %c0 : i64
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {
  // expected-error @+1 {{'func' op is not allowed inside `torch.class_type`}}
  func @f()
}

// -----

// expected-error @+1 {{has duplicate attr/method with name 'a'}}
torch.class_type @c {
  // expected-note @+1 {{see first conflicting attr/method here}}
  torch.attr "a" : i64
  // expected-note @+1 {{see second conflicting attr/method here}}
  torch.attr "a" : i64
}

// -----

torch.class_type @c {
  // expected-error @+1 {{'@invalidSym' does not reference a valid function}}
  torch.method "f", @invalidSym
}

// -----

torch.class_type @c {
  // expected-error @+1 {{'@f' must reference a private function}}
  torch.method "f", @f
}

func @f(%arg0: !torch.nn.Module<"c">) {
  return
}

// -----

torch.class_type @c {
  // expected-error @+1 {{'@f' must reference a function that is defined (not merely declared)}}
  torch.method "f", @f
}

func private @f(%arg0: !torch.nn.Module<"c">)

// -----

func private @f() {
  return
}
torch.class_type @c {
  // expected-error @+1 {{the referenced function 'f' must have a first argument of type '!torch.nn.Module<"c">'}}
  torch.method "f", @f
}

// -----

func private @f(!torch.nn.Module<"other_c">) {
  return
}
torch.class_type @c {
  // expected-error @+1 {{the referenced function 'f' must have a first argument of type '!torch.nn.Module<"c">'}}
  torch.method "f", @f
}

// -----

// expected-error @+1 {{'a' does not reference a valid class type}}
%m = torch.nn_module {} : !torch.nn.Module<"a">

// -----

// expected-error @+1 {{'torch.type_bound' must be attached to an argument of !torch.tensor/!torch.vtensor type}}
func @f(%arg0: i32 {torch.type_bound = !torch.tensor<*,f32>})

// -----

// expected-error @+1 {{'torch.type_bound' must be TypeAttr}}
func @f(%arg0: i32 {torch.type_bound = 1})

// -----

// expected-error @+1 {{'torch.type_bound' must be of !torch.tensor/!torch.vtensor type}}
func @f(%arg0: i32 {torch.type_bound = i32})

// -----

func @derefine(%arg0: !torch.optional<tensor<f32>>) -> tensor<f32> {
  // expected-error @+1 {{operand type '!torch.optional<tensor<f32>>' and result type 'tensor<f32>' are cast incompatible}}
  %0 = torch.derefine %arg0 : !torch.optional<tensor<f32>> to tensor<f32>
  return %0 : tensor<f32>
}

// -----

// expected-error @+1 {{invalid dtype 'tuple<>' for !torch.tensor type}}
func private @tensor.invalid_dtype() -> !torch.tensor<*,tuple<>>

// -----

func @torch.tensor() {
  // Incompatible shape.
  // expected-error@+1 {{incompatible}}
  %0 = torch.tensor(dense<42.0> : tensor<3x2xf32>) : !torch.vtensor<[],f32>
  return
}

// -----

func @torch.tensor() {
  // Incompatible dtype.
  // expected-error@+1 {{incompatible}}
  %0 = torch.tensor(dense<42.0> : tensor<f32>) : !torch.vtensor<[],f64>
  return
}

// -----

func @torch.tensor() {
  // Incompatible type.
  // expected-error@+1 {{incompatible}}
  %0 = torch.tensor(dense<42.0> : tensor<f32>) : i1
  return
}
