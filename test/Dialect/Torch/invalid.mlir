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
  // expected-error @+1 {{'torch.slot' op is expected to match type and name of '"torch.attr"() <{name = "g", type = !torch.int}> : () -> ()}}
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

func.func private @f(%arg0: !torch.nn.Module<"other_c">) {
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
func.func private @f(%arg0: i32 {torch.type_bound = !torch.tensor<*,f32>})

// -----

// expected-error @+1 {{'torch.type_bound' must be TypeAttr}}
func.func private @f(%arg0: i32 {torch.type_bound = 1})

// -----

// expected-error @+1 {{'torch.type_bound' must be of !torch.tensor/!torch.vtensor type}}
func.func private @f(%arg0: i32 {torch.type_bound = i32})

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

// -----

// There must be only one module initialize.

torch.global_slot.module_initializer {
  torch.initialize.global_slots [
  ]
}

// expected-error @+1 {{there must be only one global slot initializer}}
torch.global_slot.module_initializer {
  torch.initialize.global_slots [
  ]
}

// -----

// Initialized slot missing, or or non-existent slots initialized.

// expected-note @+1 {{missing global slot initializer for @slot0}}
torch.global_slot @slot0 : !torch.int
// expected-note @+1 {{missing global slot initializer for @slot1}}
torch.global_slot @slot1 : !torch.int

torch.global_slot.module_initializer {
  %0 = torch.constant.int 1
  %1 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor
  %2 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor<[],unk>
  // expected-error @below {{must have one initializer for each global slot in the module}}
  // expected-note @below {{unexpected global slot initializer for non-existent global slot @nonexistent_slot0}}
  // expected-note @below {{unexpected global slot initializer for non-existent global slot @nonexistent_slot1}}
  torch.initialize.global_slots [
    @nonexistent_slot0(%0 : !torch.int)
    @nonexistent_slot1(%0 : !torch.int)
  ]
}

// -----

// Duplicate initialization of global slot.

torch.global_slot @slot0 : !torch.int

torch.global_slot.module_initializer {
  %0 = torch.constant.int 1
  // expected-error @+1 {{duplicate initialization of global slot: @slot0}}
  torch.initialize.global_slots [
    @slot0(%0 : !torch.int)
    @slot0(%0 : !torch.int)
  ]
}

// -----

// Subtyping checks.

torch.global_slot @tensor : !torch.tensor
torch.global_slot @initialized_with_refined : !torch.tensor
torch.global_slot @error_initialized_with_derefined : !torch.tensor<[],unk>

torch.global_slot.module_initializer {
  %1 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor
  %2 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor<[],unk>
  // expected-error @below {{initial value for global slot @error_initialized_with_derefined has type '!torch.tensor' which is not within the bound '!torch.tensor<[],unk>'}}
  torch.initialize.global_slots [
    @tensor(%1 : !torch.tensor)
    @initialized_with_refined(%2 : !torch.tensor<[],unk>)
    @error_initialized_with_derefined(%1 : !torch.tensor)
  ]
}

// -----

// Restricted set of ops in the module initializer.

torch.global_slot @tensor : !torch.tensor

torch.global_slot.module_initializer {
  %0 = torch.tensor.literal(dense<0.0> : tensor<f32>) : !torch.tensor
  // expected-error @+1 {{'torch.aten.mul.Tensor' op is not allowed in a module initializer}}
  %1 = torch.aten.mul.Tensor %0, %0 : !torch.tensor, !torch.tensor -> !torch.tensor
  torch.initialize.global_slots [
    @tensor(%1 : !torch.tensor)
  ]
}

// -----

func.func @torch.tensor_static_info_cast$shape_mismatch(%arg0: !torch.vtensor<[],unk>) -> !torch.vtensor<[?],unk> {
  // expected-error@+1 {{'torch.tensor_static_info_cast' op operand type '!torch.vtensor<[],unk>' and result type '!torch.vtensor<[?],unk>' are cast incompatible}}
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[],unk> to !torch.vtensor<[?],unk>
  return %0 : !torch.vtensor<[?],unk>
}

// -----

func.func @torch.tensor_static_info_cast$dtype_mismatch(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor<*,f64> {
  // expected-error@+1 {{'torch.tensor_static_info_cast' op operand type '!torch.vtensor<*,f32>' and result type '!torch.vtensor<*,f64>' are cast incompatible}}
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<*,f32> to !torch.vtensor<*,f64>
  return %0 : !torch.vtensor<*,f64>
}


// -----

func.func @torch.permute$test_changing_rank (%arg0: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[1,2,3,4],f32> {

  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2

  %perm = torch.prim.ListConstruct %int1, %int2, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>

  // expected-error@+1 {{expected input and output tensors to have same rank, but 3 != 4}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3,4],f32>

   return %3 : !torch.vtensor<[1,2,3,4],f32>
}

// -----

func.func @torch.permute$test_permutation_too_short (%arg0: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[1,2,3],f32> {

  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1

  %perm = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>

  // expected-error@+1 {{The permutation has 2 elements, the output has rank 3}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3],f32>

   return %3 : !torch.vtensor<[1,2,3],f32>
}

// -----

func.func @torch.permute$duplicate_index_in_permutation (%arg0: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[2,3,1],f32> {

  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %perm = torch.prim.ListConstruct %int1, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>

  // expected-error@+1 {{'torch.aten.permute' op has a duplicate dimension (1) in its permutation}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,3,1],f32>

   return %3 : !torch.vtensor<[2,3,1],f32>
}

// -----

func.func @torch.permute$incorrect_output_shape (%arg0: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[3,1,2],f32> {

  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %none = torch.constant.none

  %perm = torch.prim.ListConstruct %int1, %int2, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>

  // expected-error@+1 {{'torch.aten.permute' op has a permutation which is not compatible with the input and output shapes. The input shape in dimension 1 is 2, and the output shape in dimension 0 is 3 : they should be the same with this permutation.}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[3,1,2],f32>

   return %3 : !torch.vtensor<[3,1,2],f32>
}


// -----

func.func @torch.permute$invalid_index_in_permutation (%arg0: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[1,2,3],f32> {

  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int7 = torch.constant.int 7
  %perm = torch.prim.ListConstruct %int0, %int1, %int7 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>


  // expected-error@+1 {{observed invalid index in permutation (7) for input tensor of rank 3.}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3],f32>

   return %3 : !torch.vtensor<[1,2,3],f32>
}

// -----

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// expected-error @+1 {{dimension-rank mismatch between encoding and tensor shape: 1 != 2}}
func.func @foo(%arg0: !torch.vtensor<[64,64],f32,#SV>) -> !torch.vtensor<[64,64],f32,#SV> {
  return %arg0 : !torch.vtensor<[64,64],f32,#SV>
}

// -----

// expected-error @+1 {{invalid sparsity encoding attribute}}
func.func private @tensor.sparse() -> !torch.vtensor<[64,64],f32,12345>
