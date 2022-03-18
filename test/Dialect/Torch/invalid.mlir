// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics

// -----

torch.class_type @c {}
%0 = torch.nn_module {
  // expected-error @+1 {{'func.func' op is not allowed inside 'torch.nn_module'}}
  func.func @f()
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {}
%c0 = torch.constant.int 0
// expected-error @+1 {{number of 'torch.slot's in a 'torch.nn_module' must match number of 'torch.attr's in the corresponding 'torch.class_type'}}
%0 = torch.nn_module {
  torch.slot "f", %c0 : !torch.int
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {
  // expected-note @+1 {{see torch.attr at corresponding index 0 here}}
  torch.attr "g" : !torch.int
}
%c0 = torch.constant.int 0
%0 = torch.nn_module {
  // expected-error @+1 {{'torch.slot' op is expected to match type and name of '"torch.attr"() {name = "g", type = !torch.int} : () -> ()}}
  torch.slot "f", %c0 : !torch.int
} : !torch.nn.Module<"c">

// -----

torch.class_type @c {
  // expected-error @+1 {{'func.func' op is not allowed inside `torch.class_type`}}
  func.func @f()
}

// -----

// expected-error @+1 {{has duplicate attr/method with name 'a'}}
torch.class_type @c {
  // expected-note @+1 {{see first conflicting attr/method here}}
  torch.attr "a" : !torch.int
  // expected-note @+1 {{see second conflicting attr/method here}}
  torch.attr "a" : !torch.int
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

func.func @f(%arg0: !torch.nn.Module<"c">) {
  return
}

// -----

torch.class_type @c {
  // expected-error @+1 {{'@f' must reference a function that is defined (not merely declared)}}
  torch.method "f", @f
}

func.func private @f(%arg0: !torch.nn.Module<"c">)

// -----

func.func private @f() {
  return
}
torch.class_type @c {
  // expected-error @+1 {{the referenced function 'f' must have a first argument of type '!torch.nn.Module<"c">'}}
  torch.method "f", @f
}

// -----

func.func private @f(!torch.nn.Module<"other_c">) {
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
func.func @f(%arg0: i32 {torch.type_bound = !torch.tensor<*,f32>})

// -----

// expected-error @+1 {{'torch.type_bound' must be TypeAttr}}
func.func @f(%arg0: i32 {torch.type_bound = 1})

// -----

// expected-error @+1 {{'torch.type_bound' must be of !torch.tensor/!torch.vtensor type}}
func.func @f(%arg0: i32 {torch.type_bound = i32})

// -----

func.func @derefine(%arg0: !torch.optional<tensor>) -> !torch.tensor {
  // expected-error @+1 {{operand type '!torch.optional<tensor>' and result type '!torch.tensor' are cast incompatible}}
  %0 = torch.derefine %arg0 : !torch.optional<tensor> to !torch.tensor
  return %0 : !torch.tensor
}

// -----

func.func @torch.prim.unchecked_cast$invalid_types(%arg0: !torch.tensor) -> !torch.optional<tensor> {
  // expected-error @+1 {{operand type '!torch.tensor' and result type '!torch.optional<tensor>' are cast incompatible}}
  %0 = torch.prim.unchecked_cast %arg0 : !torch.tensor -> !torch.optional<tensor>
  return %0 : !torch.optional<tensor>
}

// -----

// expected-error @+1 {{invalid dtype 'tuple<>' for !torch.tensor type}}
func.func private @tensor.invalid_dtype() -> !torch.tensor<*,tuple<>>

// -----

func.func @torch.tensor() {
  // Incompatible shape.
  // expected-error@+1 {{must be Multi-dimensional array modeling Torch's Tensor type, but got}}
  %0 = torch.tensor.literal(dense<42.0> : tensor<3x2xf32>) : !torch.vtensor<[],f32>
  return
}

// -----

func.func @torch.tensor() {
  // Incompatible dtype.
  // expected-error@+1 {{must be Multi-dimensional array modeling Torch's Tensor type, but got}}
  %0 = torch.tensor.literal(dense<42.0> : tensor<f32>) : !torch.vtensor<[],f64>
  return
}

// -----

func.func @torch.tensor() {
  // Incompatible type.
  // expected-error@+1 {{must be Multi-dimensional array modeling Torch's Tensor type, but got}}
  %0 = torch.tensor.literal(dense<42.0> : tensor<f32>) : i1
  return
}

// -----

func.func @torch.prim.ListConstruct() {
  %int2 = torch.constant.int 2
  // expected-error@+1 {{operand types should have the same type as the list contained type}}
  torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<tensor>
  return
}

// -----

func.func @torch.overwrite.tensor.contents(%arg0: !torch.vtensor<[1],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[1],f32> {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor<[1],f32>
  // expected-error@+1 {{'torch.overwrite.tensor.contents' op failed to verify that overwritten tensor type is corresponding !torch.tensor of value tensor type}}
  torch.overwrite.tensor.contents %arg1 overwrites %0 : !torch.vtensor<[?],f32>, !torch.tensor<[1],f32>
  %1 = torch.copy.to_vtensor %0 : !torch.vtensor<[1],f32>
  return %1 : !torch.vtensor<[1],f32>
}
