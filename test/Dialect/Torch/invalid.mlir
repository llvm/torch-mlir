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

// expected-error @+1 {{there must be only one global slot initializer}}
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

func.func @torch.permute$duplicate_negative_index_in_permutation (%arg0: !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,2,4,4],f32> {

  %intm1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %perm = torch.prim.ListConstruct %int0, %int1, %int3, %intm1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>

  // expected-error@+1 {{'torch.aten.permute' op has a duplicate dimension (3) in its permutation}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,4,4],f32>

  return %3 : !torch.vtensor<[1,2,4,4],f32>
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

func.func @torch.permute$invalid_negative_index_in_permutation (%arg0: !torch.vtensor<[1,2,3,4],f32>) -> !torch.vtensor<[1,2,3,4],f32> {

  %intm5 = torch.constant.int -5
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %perm = torch.prim.ListConstruct %int0, %int1, %int2, %intm5 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>

  // expected-error@+1 {{observed invalid index in permutation (-5) for input tensor of rank 4.}}
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3,4],f32>

  return %3 : !torch.vtensor<[1,2,3,4],f32>
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


// -----

func.func @torch.symbolic_int$no_shape_symbols(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %0 = torch.symbolic_int "s0" {min_val = 3, max_val = 6} : !torch.int
  // expected-error @+1 {{op requires equal number of shape symbol args and symbol args to the attached affine map, since they are 1:1 mapped}}
  torch.bind_symbolic_shape %arg0, [], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
  return %arg0 : !torch.vtensor<[?],f32>
}

// -----

// Verifier should not fail here since the op does not require shapeSymbols.
func.func @torch.symbolic_int$no_shape_symbols_no_symbols_in_map(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  torch.bind_symbolic_shape %arg0, [], affine_map<()[] -> (1)> : !torch.vtensor<[?],f32>
  return %arg0 : !torch.vtensor<[?],f32>
}

// -----

func.func @torch.symbolic_int$no_shape_symbols(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
  %int0 = torch.constant.int 0
  // expected-error @+1 {{shape symbol must be produced by a SymbolicIntOp}}
  torch.bind_symbolic_shape %arg0, [%int0], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
  return %arg0 : !torch.vtensor<[?],f32>
}

// -----

func.func @torch.aten._scaled_mm$invalid_blockwise_scale_numel(
    %arg0: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg2: !torch.vtensor<[512],f8E4M3FN>,
    %arg3: !torch.vtensor<[512],f8E4M3FN>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op invalid blockwise scaling configuration: expected scale_a to have 2048 elements and scale_b to have 2048 elements, but got 512 and 512}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %false : !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[512],f8E4M3FN>, !torch.vtensor<[512],f8E4M3FN>, !torch.none, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm$invalid_self_dtype(
    %arg0: !torch.vtensor<[128,128],f32>,
    %arg1: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op expected self to have an FP8 or FP4 dtype}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %false : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[512],f8E8M0FNU>, !torch.vtensor<[512],f8E8M0FNU>, !torch.none, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm$invalid_mat2_non_contracting_dim(
    %arg0: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[128,127],f8E4M3FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,127],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op expected mat2 non-contracting dimension to be divisible by 16, but got 127}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %false : !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[128,127],f8E4M3FN>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[128,127],bf16>
  return %0 : !torch.vtensor<[128,127],bf16>
}

// -----

func.func @torch.aten._scaled_mm$invalid_mixed_scale_dtype(
    %arg0: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op expected non-tensorwise, non-blockwise scale_a and scale_b to have f32 dtype}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %false : !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[512],f8E8M0FNU>, !torch.vtensor<[512],f32>, !torch.none, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm$invalid_fp4_blockwise_scale_numel(
    %arg0: !torch.vtensor<[128,128],f4E2M1FN>,
    %arg1: !torch.vtensor<[128,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op invalid blockwise scaling configuration: expected scale_a to have 1024 elements and scale_b to have 1024 elements, but got 512 and 512}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %none, %none, %int15, %false : !torch.vtensor<[128,128],f4E2M1FN>, !torch.vtensor<[128,128],f4E2M1FN>, !torch.vtensor<[512],f8E8M0FNU>, !torch.vtensor<[512],f8E8M0FNU>, !torch.none, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_mxfp4_blockwise_scale_numel(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[256],f8E8M0FNU>,
    %arg3: !torch.vtensor<[256],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[256],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[256],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{invalid blockwise scaling configuration: expected scale_a to have 512 elements and scale_b to have 512 elements, but got 256 and 256}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_mxfp4_blockwise_swizzle(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected blockwise swizzle_a and swizzle_b to be SWIZZLE_32_4_4}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_unsupported_recipe(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[128,1],f32>,
    %arg3: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int5 = torch.constant.int 5
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[128,1],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int5 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[1,128],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int5 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{invalid scaling configuration for recipe_a and recipe_b}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_static_recipe_dynamic_swizzle(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[128,1],f32>,
    %arg3: !torch.vtensor<[1,128],f32>,
    %arg4: !torch.list<int>,
    %arg5: !torch.list<int>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int5 = torch.constant.int 5
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[128,1],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int5 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[1,128],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int5 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{invalid scaling configuration for recipe_a and recipe_b}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %arg4, %scale_b, %recipe_b, %arg5, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_f32_blockwise_1x128_1x128_scale_shape(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[128,1],f32>,
    %arg3: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int4 = torch.constant.int 4
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[128,1],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[1,128],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{invalid 1x128 x 1x128 blockwise scaling configuration}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_empty_scale_recipe_lists(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct : () -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct : () -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct : () -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct : () -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct : () -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct : () -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected scale_a, recipe_a, scale_b and recipe_b lists to be non-empty}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_two_level_nv_tensorwise_scale_numel(
    %arg0: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg2: !torch.vtensor<[1024],f8E4M3FN>,
    %arg3: !torch.vtensor<[2],f32>,
    %arg4: !torch.vtensor<[1024],f8E4M3FN>,
    %arg5: !torch.vtensor<[2],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2, %arg3 : (!torch.vtensor<[1024],f8E4M3FN>, !torch.vtensor<[2],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int2, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg4, %arg5 : (!torch.vtensor<[1024],f8E4M3FN>, !torch.vtensor<[2],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int2, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected two-level NV tensorwise scale_a and scale_b to both be scalar}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_self_dtype(
    %arg0: !torch.vtensor<[128,64],f32>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected self to have an FP8 or FP4 dtype}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f32>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_mat2_dtype(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f32>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected mat2 to have an FP8 or FP4 dtype}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f32>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_rank(
    %arg0: !torch.vtensor<[1,128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected self and mat2 to be rank 2, but got ranks 3 and 2}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[1,128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_contracting_dim_mismatch(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[65,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected self and mat2 contracting dimensions to match, but got 128 and 130}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[65,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_contraction_dim_length(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  // expected-error @+1 {{contraction_dim must have exactly 2 elements}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_contraction_dim_mismatch(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{expected self and mat2 contraction_dim-selected dimensions to match, but got 128 and 64}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_mat2_non_contracting_dim_divisibility(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,127],f4E2M1FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[],f32>) -> !torch.vtensor<[128,127],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected mat2 non-contracting dimension to be divisible by 16, but got 127}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,127],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,127],bf16>
  return %0 : !torch.vtensor<[128,127],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_contracting_dim_divisibility(
    %arg0: !torch.vtensor<[128,63],f4E2M1FN>,
    %arg1: !torch.vtensor<[63,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected self contracting dimension to be divisible by 16, but got 126}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,63],f4E2M1FN>, !torch.vtensor<[63,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_tensorwise_scale_numel(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[1,128],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected scale_a and scale_b to both be scalar for tensorwise scaling}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_tensorwise_scale_dtype(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[],f8E8M0FNU>,
    %arg3: !torch.vtensor<[],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected tensorwise scale_a and scale_b to have f32 dtype}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_mixed_scale_dtype(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[128,1],f8E8M0FNU>,
    %arg3: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[128,1],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[1,128],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected rowwise scale_a and scale_b to have f32 dtype}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_recipe_list_length(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int3, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{expected scale_a and recipe_a lists to have the same length}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm_v2$invalid_rowwise_scale_shape(
    %arg0: !torch.vtensor<[128,64],f4E2M1FN>,
    %arg1: !torch.vtensor<[64,128],f4E2M1FN>,
    %arg2: !torch.vtensor<[64,2],f32>,
    %arg3: !torch.vtensor<[2,64],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  %scale_a = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[64,2],f32>) -> !torch.list<vtensor>
  %recipe_a = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %swizzle_a = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %scale_b = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[2,64],f32>) -> !torch.list<vtensor>
  %recipe_b = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %swizzle_b = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %contraction_dim = torch.prim.ListConstruct : () -> !torch.list<int>
  // expected-error @+1 {{invalid rowwise scaling configuration}}
  %0 = torch.aten._scaled_mm_v2 %arg0, %arg1, %scale_a, %recipe_a, %swizzle_a, %scale_b, %recipe_b, %swizzle_b, %none, %int15, %contraction_dim, %false : !torch.vtensor<[128,64],f4E2M1FN>, !torch.vtensor<[64,128],f4E2M1FN>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm$invalid_bias_dtype(
    %arg0: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[],f32>,
    %arg4: !torch.vtensor<[128],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op expected bias to have bf16 or f16 dtype, but got 'f32'}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %arg4, %none, %int15, %false : !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[128],f32>, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.aten._scaled_mm$invalid_scale_result_numel(
    %arg0: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg1: !torch.vtensor<[128,128],f8E4M3FN>,
    %arg2: !torch.vtensor<[],f32>,
    %arg3: !torch.vtensor<[],f32>,
    %arg4: !torch.vtensor<[2],f32>) -> !torch.vtensor<[128,128],bf16> {
  %false = torch.constant.bool false
  %int15 = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.aten._scaled_mm' op expected scale_result to have 1 element, but got 2}}
  %0 = torch.aten._scaled_mm %arg0, %arg1, %arg2, %arg3, %none, %arg4, %int15, %false : !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[128,128],f8E4M3FN>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.none, !torch.vtensor<[2],f32>, !torch.int, !torch.bool -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

func.func @torch.torchao.dequantize_affine$invalid_zero_point_numel(
    %arg0: !torch.vtensor<[4,16],si8>,
    %arg1: !torch.vtensor<[4,2],f32>,
    %arg2: !torch.vtensor<[4,4],si32>) -> !torch.vtensor<[4,16],f32> {
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.dequantize_affine' op zero_point must have as many elements as scale}}
  %0 = torch.torchao.dequantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none, %int6 : !torch.vtensor<[4,16],si8>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,4],si32>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.torchao.quantize_affine$invalid_block_size_length(
    %arg0: !torch.vtensor<[4,16],f32>,
    %arg1: !torch.vtensor<[4,2],f32>,
    %arg2: !torch.vtensor<[4,2],si32>) -> !torch.vtensor<[4,16],si8> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int8 : (!torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.quantize_affine' op block_size must have one entry per input dimension}}
  %0 = torch.torchao.quantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none : !torch.vtensor<[4,16],f32>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],si32>, !torch.int, !torch.none, !torch.none -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.torchao.quantize_affine$invalid_block_size_divisor(
    %arg0: !torch.vtensor<[?,16],f32>,
    %arg1: !torch.vtensor<[1,3],f32>,
    %arg2: !torch.vtensor<[1,3],si32>) -> !torch.vtensor<[?,16],si8> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int5 = torch.constant.int 5
  %none = torch.constant.none
  %size = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,16],f32>, !torch.int -> !torch.int
  %block_size = torch.prim.ListConstruct %size, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.quantize_affine' op block_size entries must be positive divisors of the input sizes}}
  %0 = torch.torchao.quantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none : !torch.vtensor<[?,16],f32>, !torch.list<int>, !torch.vtensor<[1,3],f32>, !torch.vtensor<[1,3],si32>, !torch.int, !torch.none, !torch.none -> !torch.vtensor<[?,16],si8>
  return %0 : !torch.vtensor<[?,16],si8>
}

// -----

func.func @torch.torchao.quantize_affine$invalid_scale_numel(
    %arg0: !torch.vtensor<[4,16],f32>,
    %arg1: !torch.vtensor<[4,4],f32>,
    %arg2: !torch.vtensor<[4,4],si32>) -> !torch.vtensor<[4,16],si8> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.quantize_affine' op scale must have one element per block}}
  %0 = torch.torchao.quantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none : !torch.vtensor<[4,16],f32>, !torch.list<int>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],si32>, !torch.int, !torch.none, !torch.none -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.torchao.quantize_affine$invalid_scale_dtype(
    %arg0: !torch.vtensor<[4,16],f32>,
    %arg1: !torch.vtensor<[4,2],f8E4M3FN>,
    %arg2: !torch.vtensor<[4,2],si32>) -> !torch.vtensor<[4,16],si8> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.quantize_affine' op scale must have an f16, bf16, f32 or f64 dtype}}
  %0 = torch.torchao.quantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none : !torch.vtensor<[4,16],f32>, !torch.list<int>, !torch.vtensor<[4,2],f8E4M3FN>, !torch.vtensor<[4,2],si32>, !torch.int, !torch.none, !torch.none -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.torchao.dequantize_affine$invalid_zero_point_dtype(
    %arg0: !torch.vtensor<[4,16],si8>,
    %arg1: !torch.vtensor<[4,2],f32>,
    %arg2: !torch.vtensor<[4,2],f8E4M3FN>) -> !torch.vtensor<[4,16],f32> {
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.dequantize_affine' op zero_point must have an integer, f16, bf16, f32 or f64 dtype}}
  %0 = torch.torchao.dequantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none, %int6 : !torch.vtensor<[4,16],si8>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f8E4M3FN>, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.torchao.dequantize_affine$invalid_block_size_zero(
    %arg0: !torch.vtensor<[4,16],si8>,
    %arg1: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[4,16],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.dequantize_affine' op block_size entries must be positive divisors of the input sizes}}
  %0 = torch.torchao.dequantize_affine %arg0, %block_size, %arg1, %none, %int1, %none, %none, %int6 : !torch.vtensor<[4,16],si8>, !torch.list<int>, !torch.vtensor<[4,1],f32>, !torch.none, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.torchao.quantize_affine$invalid_quant_max(
    %arg0: !torch.vtensor<[4,16],f32>,
    %arg1: !torch.vtensor<[4,2],f32>,
    %arg2: !torch.vtensor<[4,2],si32>) -> !torch.vtensor<[4,16],si8> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %int200 = torch.constant.int 200
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.quantize_affine' op quant_min and quant_max must lie within the bounds of output_dtype}}
  %0 = torch.torchao.quantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %int200 : !torch.vtensor<[4,16],f32>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],si32>, !torch.int, !torch.none, !torch.int -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.torchao.quantize_affine$invalid_input_dtype(
    %arg0: !torch.vtensor<[4,16],f64>,
    %arg1: !torch.vtensor<[4,2],f32>,
    %arg2: !torch.vtensor<[4,2],si32>) -> !torch.vtensor<[4,16],si8> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.quantize_affine' op input must have an f16, bf16 or f32 dtype}}
  %0 = torch.torchao.quantize_affine %arg0, %block_size, %arg1, %arg2, %int1, %none, %none : !torch.vtensor<[4,16],f64>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],si32>, !torch.int, !torch.none, !torch.none -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.torchao.dequantize_affine$invalid_input_dtype(
    %arg0: !torch.vtensor<[4,16],si8>,
    %arg1: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,16],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.dequantize_affine' op input_dtype must be the dtype of the quantized tensor}}
  %0 = torch.torchao.dequantize_affine %arg0, %block_size, %arg1, %none, %int0, %none, %none, %int6 : !torch.vtensor<[4,16],si8>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.none, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.torchao.dequantize_affine$unsupported_input_dtype(
    %arg0: !torch.vtensor<[4,16],si64>,
    %arg1: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,16],f32> {
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int6 = torch.constant.int 6
  %int8 = torch.constant.int 8
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{'torch.torchao.dequantize_affine' op input_dtype must be uint8, int8, int16, int32 or a float8 dtype}}
  %0 = torch.torchao.dequantize_affine %arg0, %block_size, %arg1, %none, %int4, %none, %none, %int6 : !torch.vtensor<[4,16],si64>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.none, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_tensor$invalid_real_dtype(%arg0: !torch.vtensor<[4,8],si32>) -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_tensor' op expected input to have a floating-point dtype, but got 'si32'}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],si32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_tensor$invalid_dtype_code(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 6
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_tensor' op expected dtype to be a PT2E storage dtype, but got dtype code 6}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_tensor$invalid_bound_order(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int 127
  %quant_max = torch.constant.int -128
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_tensor' op expected quant_min to not exceed quant_max, but got [127, -128]}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_tensor$invalid_quant_min(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],ui8> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 0
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_tensor' op expected quant_min to be representable in 'ui8', but got -128}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],ui8>
  return %0 : !torch.vtensor<[4,8],ui8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_tensor$invalid_quant_max(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 255
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_tensor' op expected quant_max to be representable in 'si8', but got 255}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_tensor$invalid_float8_quant_max(%arg0: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[4,8],f8E4M3FN> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -448
  %quant_max = torch.constant.int 449
  %dtype = torch.constant.int 24
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_tensor' op expected quant_max to be representable in 'f8E4M3FN', but got 449}}
  %0 = torch.quantized_decomposed.quantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],f8E4M3FN>
  return %0 : !torch.vtensor<[4,8],f8E4M3FN>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_tensor$invalid_dtype_mismatch(%arg0: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 2
  %out_dtype = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_tensor' op expected dtype code 2 ('si16') to match the input dtype 'si8'}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_tensor$invalid_storage_dtype(%arg0: !torch.vtensor<[4,8],si64>, %dtype: !torch.int) -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %out_dtype = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_tensor' op expected input to have a PT2E storage dtype, but got 'si64'}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si64>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_tensor$invalid_result_dtype(%arg0: !torch.vtensor<[4,8],si8>, %out_dtype: !torch.int) -> !torch.vtensor<[4,8],si64> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_tensor' op expected result to have a floating-point dtype, but got 'si64'}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si64>
  return %0 : !torch.vtensor<[4,8],si64>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_tensor$invalid_out_dtype_int(%arg0: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 4
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_tensor' op expected out_dtype to be a floating-point dtype, but got dtype code 4}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_tensor$invalid_out_dtype_mismatch(%arg0: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 15
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_tensor' op expected the result dtype to be 'bf16', but got 'f32'}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_tensor$invalid_default_out_dtype(%arg0: !torch.vtensor<[4,8],si8>) -> !torch.vtensor<[4,8],f16> {
  %scale = torch.constant.float 5.000000e-01
  %zero_point = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_tensor' op expected the result dtype to be 'f32', but got 'f16'}}
  %0 = torch.quantized_decomposed.dequantize_per_tensor %arg0, %scale, %zero_point, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f16>
  return %0 : !torch.vtensor<[4,8],f16>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel$invalid_scales_dtype(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[4],si64>, %arg2: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,8],si8> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel' op expected scales to have a floating-point dtype, but got 'si64'}}
  %0 = torch.quantized_decomposed.quantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel$invalid_scales_rank(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[4,1],f32>, %arg2: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,8],si8> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel' op expected scales to have rank 1, but got rank 2}}
  %0 = torch.quantized_decomposed.quantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4,1],f32>, !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel$invalid_scale_broadcast(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,8],si8> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel' op expected scales to have one element per index of input dimension 0 (4), but got 1}}
  %0 = torch.quantized_decomposed.quantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel$invalid_zero_points_rank(%arg0: !torch.vtensor<[4,8],si8>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4,1],si64>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel' op expected zero_points to have rank 1, but got rank 2}}
  %0 = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype, %none : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel$invalid_dtype_code(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,8],si8> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 2
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel' op expected dtype code 2 ('si16') to match the result dtype 'si8'}}
  %0 = torch.quantized_decomposed.quantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel$invalid_out_dtype(%arg0: !torch.vtensor<[4,8],si8>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %out_dtype = torch.constant.int 5
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel' op expected the result dtype to be 'f16', but got 'f32'}}
  %0 = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg1, %none, %axis, %quant_min, %quant_max, %dtype, %out_dtype : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel$invalid_zero_points_length(%arg0: !torch.vtensor<[4,8],si8>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[3],si64>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel' op expected zero_points to have the length of scales, but got 3 and 4}}
  %0 = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype, %none : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.vtensor<[3],si64>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel$invalid_axis(%arg0: !torch.vtensor<[4,8],si8>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 2
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel' op expected axis to be in [-2, 2), but got 2}}
  %0 = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype, %none : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel$invalid_negative_axis(%arg0: !torch.vtensor<[4,8],si8>, %arg1: !torch.vtensor<[4],f32>, %arg2: !torch.none) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int -3
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel' op expected axis to be in [-2, 2), but got -3}}
  %0 = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype, %none : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel$invalid_scales_length(%arg0: !torch.vtensor<[4,8],si8>, %arg1: !torch.vtensor<[8],f32>, %arg2: !torch.none) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %quant_min = torch.constant.int -128
  %quant_max = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel' op expected scales to have one element per index of input dimension 0 (4), but got 8}}
  %0 = torch.quantized_decomposed.dequantize_per_channel %arg0, %arg1, %arg2, %axis, %quant_min, %quant_max, %dtype, %none : !torch.vtensor<[4,8],si8>, !torch.vtensor<[8],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none -> !torch.vtensor<[4,8],f32>
  return %0 : !torch.vtensor<[4,8],f32>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_input_rank(%arg0: !torch.vtensor<[2,4,16],f32>, %arg1: !torch.vtensor<[2,4],f32>, %arg2: !torch.vtensor<[2,4],si64>) -> !torch.vtensor<[2,4,16],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected input to have rank 2, but got rank 3}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[2,4,16],f32>, !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4,16],si8>
  return %0 : !torch.vtensor<[2,4,16],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_scales_dtype(%arg0: !torch.vtensor<[4,16],f32>, %arg1: !torch.vtensor<[4,4],si64>, %arg2: !torch.vtensor<[4,4],si64>) -> !torch.vtensor<[4,16],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected scales to have a floating-point dtype, but got 'si64'}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,4],si64>, !torch.vtensor<[4,4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_zero_row_length(%arg0: !torch.vtensor<[4,0],f32>, %arg1: !torch.vtensor<[4,1],f32>, %arg2: !torch.vtensor<[4,1],si64>) -> !torch.vtensor<[4,0],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 32
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected input dimension 1 to be non-zero when group_size exceeds it and the last dimension of scales is 1}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,0],f32>, !torch.vtensor<[4,1],f32>, !torch.vtensor<[4,1],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,0],si8>
  return %0 : !torch.vtensor<[4,0],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_zero_points_count(%arg0: !torch.vtensor<[4,16],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,16],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected zero_points to have as many elements as scales, but got 8 and 16}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_group_size(%arg0: !torch.vtensor<[4,16],f32>, %arg1: !torch.vtensor<[4,16],f32>, %arg2: !torch.vtensor<[4,16],si64>) -> !torch.vtensor<[4,16],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 1
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected group_size to be greater than 1, but got 1}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_zero_points_group_count(%arg0: !torch.vtensor<[4,16],f32>, %arg1: !torch.vtensor<[?,4],f32>, %arg2: !torch.vtensor<[2,4],si64>) -> !torch.vtensor<[4,16],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected zero_points to have 16 elements, one per group, but got 8}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,16],f32>, !torch.vtensor<[?,4],f32>, !torch.vtensor<[2,4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_gptq_zero_points_count(%arg0: !torch.vtensor<[4,8],f32>, %arg1: !torch.vtensor<[4,?],f32>, %arg2: !torch.vtensor<[4,2],si64>) -> !torch.vtensor<[4,8],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 16
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected zero_points to have 4 elements, one per group, but got 8}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4,?],f32>, !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %0 : !torch.vtensor<[4,8],si8>
}

// -----

func.func @torch.quantized_decomposed.quantize_per_channel_group$invalid_quant_max(%arg0: !torch.vtensor<[4,16],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4,4],si64>) -> !torch.vtensor<[4,16],si8> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 128
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  // expected-error @+1 {{'torch.quantized_decomposed.quantize_per_channel_group' op expected quant_max to be representable in 'si8', but got 128}}
  %0 = torch.quantized_decomposed.quantize_per_channel_group %arg0, %arg1, %arg2, %quant_min, %quant_max, %dtype, %group_size : !torch.vtensor<[4,16],f32>, !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],si8>
  return %0 : !torch.vtensor<[4,16],si8>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel_group$invalid_divisibility(%arg0: !torch.vtensor<[4,16],si8>, %arg1: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[4,16],f32> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 5
  %output_dtype = torch.constant.int 6
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel_group' op expected input dimension 1 (16) to be divisible by group_size 5}}
  %0 = torch.quantized_decomposed.dequantize_per_channel_group %arg0, %arg1, %none, %quant_min, %quant_max, %dtype, %group_size, %output_dtype : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,3],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel_group$invalid_group_count(%arg0: !torch.vtensor<[4,16],si8>, %arg1: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,16],f32> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  %output_dtype = torch.constant.int 6
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel_group' op expected scales to have 16 elements, one per group, but got 8}}
  %0 = torch.quantized_decomposed.dequantize_per_channel_group %arg0, %arg1, %none, %quant_min, %quant_max, %dtype, %group_size, %output_dtype : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,2],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel_group$invalid_gptq_multiple_scale_columns(%arg0: !torch.vtensor<[4,16],si8>, %arg1: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,16],f32> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 128
  %output_dtype = torch.constant.int 6
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel_group' op expected input dimension 1 (16) to be divisible by group_size 128}}
  %0 = torch.quantized_decomposed.dequantize_per_channel_group %arg0, %arg1, %none, %quant_min, %quant_max, %dtype, %group_size, %output_dtype : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,2],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}

// -----

func.func @torch.quantized_decomposed.dequantize_per_channel_group$invalid_out_dtype(%arg0: !torch.vtensor<[4,16],si8>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,16],f32> {
  %quant_min = torch.constant.int -8
  %quant_max = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  %output_dtype = torch.constant.int 15
  %none = torch.constant.none
  // expected-error @+1 {{'torch.quantized_decomposed.dequantize_per_channel_group' op expected the result dtype to be 'bf16', but got 'f32'}}
  %0 = torch.quantized_decomposed.dequantize_per_channel_group %arg0, %arg1, %none, %quant_min, %quant_max, %dtype, %group_size, %output_dtype : !torch.vtensor<[4,16],si8>, !torch.vtensor<[4,4],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,16],f32>
  return %0 : !torch.vtensor<[4,16],f32>
}
