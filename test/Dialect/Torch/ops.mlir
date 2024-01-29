// RUN: torch-mlir-opt %s | torch-mlir-opt | FileCheck %s

// CHECK: #[[$ENCODING:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK-LABEL: func.func @torch.operator(
func.func @torch.operator(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor {
  // CHECK: torch.operator "ns.unqual.overload"(%arg0, %arg1) : (!torch.tensor, !torch.tensor) -> !torch.tensor
  %0 = torch.operator "ns.unqual.overload"(%arg0, %arg1) : (!torch.tensor, !torch.tensor) -> !torch.tensor
  return %0 : !torch.tensor
}

func.func @torch.linear_params.create(%arg0: !torch.tensor, %arg1: !torch.tensor) -> (!torch.LinearParams, !torch.LinearParams) {
  %with_bias = torch.linear_params.create %arg0, %arg1 : !torch.tensor, !torch.tensor
  %without_bias = torch.linear_params.create %arg0 : !torch.tensor
  return %with_bias, %without_bias : !torch.LinearParams, !torch.LinearParams
}

// CHECK: @tensor.default() -> !torch.tensor
func.func private @tensor.default() -> !torch.tensor
// CHECK: @tensor.default_explicit() -> !torch.tensor{{$}}
func.func private @tensor.default_explicit() -> !torch.tensor<*,unk>
// CHECK: @tensor.value_semantic() -> !torch.vtensor{{$}}
func.func private @tensor.value_semantic() -> !torch.vtensor<*,unk>
// CHECK: @tensor.dtype() -> !torch.tensor<*,si32>
func.func private @tensor.dtype() -> !torch.tensor<*,si32>
// CHECK: @tensor.ranked() -> !torch.tensor<[?,?,?],unk>
func.func private @tensor.ranked() -> !torch.tensor<[?,?,?],unk>
// CHECK: @tensor.some_sizes_known() -> !torch.tensor<[?,2,?,4],unk>
func.func private @tensor.some_sizes_known() -> !torch.tensor<[?,2,?,4],unk>
// CHECK: @tensor.fully_determined() -> !torch.vtensor<[1,2,3,4],f32>
func.func private @tensor.fully_determined() -> !torch.vtensor<[1,2,3,4],f32>

// CHECK: @tensor.sparse() -> !torch.vtensor<[64,64],f32,#[[$ENCODING]]>
#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func private @tensor.sparse() -> !torch.vtensor<[64,64],f32,#CSR>

// CHECK: @tuple.empty() -> !torch.tuple<>
func.func private @tuple.empty() -> !torch.tuple<>
// CHECK: @tuple.one_element() -> !torch.tuple<tensor>
func.func private @tuple.one_element() -> !torch.tuple<tensor>
// CHECK: @tuple.two_elements() -> !torch.tuple<tensor, tensor>
func.func private @tuple.two_elements() -> !torch.tuple<tensor, tensor>

// CHECK: @union.empty() -> !torch.union<>
func.func private @union.empty() -> !torch.union<>
// CHECK: @union.one_element() -> !torch.union<tensor>
func.func private @union.one_element() -> !torch.union<tensor>
// CHECK: @union.two_elements() -> !torch.union<tensor, tensor>
func.func private @union.two_elements() -> !torch.union<tensor, tensor>

// CHECK: @dict() -> !torch.dict<str, tensor>
func.func private @dict() -> !torch.dict<str, tensor>

// CHECK-LABEL:   func.func @torch.tensor.literal() {
func.func @torch.tensor.literal() {
  // CHECK: torch.tensor.literal(dense<4.200000e+01> : tensor<3x2xf32>) : !torch.tensor
  %0 = torch.tensor.literal(dense<42.0> : tensor<3x2xf32>) : !torch.tensor
  // CHECK: torch.tensor.literal(dense<4.200000e+01> : tensor<3x2xf32>) : !torch.tensor<[3,2],f32>
  %1 = torch.tensor.literal(dense<42.0> : tensor<3x2xf32>) : !torch.tensor<[3,2],f32>
  return
}

// CHECK-LABEL:   func.func @torch.vtensor.literal() {
func.func @torch.vtensor.literal() {
  // CHECK: torch.vtensor.literal(dense<4.200000e+01> : tensor<3x2xf32>) : !torch.vtensor<[3,2],f32>
  %0 = torch.vtensor.literal(dense<42.0> : tensor<3x2xf32>) : !torch.vtensor<[3,2],f32>
  return
}

func.func @derefine(%arg0: !torch.tensor) -> !torch.optional<tensor> {
  %0 = torch.derefine %arg0 : !torch.tensor to !torch.optional<tensor>
  return %0 : !torch.optional<tensor>
}

func.func @torch.prim.If(%arg0: !torch.bool, %arg1: !torch.int) -> !torch.int {
  %0 = torch.prim.If %arg0 -> (!torch.int) {
    %1 = torch.aten.add.int %arg1, %arg1 : !torch.int, !torch.int -> !torch.int
    torch.prim.If.yield %1 : !torch.int
  } else {
    %1 = torch.aten.mul.int %arg1, %arg1 : !torch.int, !torch.int -> !torch.int
    torch.prim.If.yield %1 : !torch.int
  }
  return %0 : !torch.int
}

// CHECK: %true = torch.constant.bool true
%true = torch.constant.bool true
// CHECK: %false = torch.constant.bool false
%false = torch.constant.bool false

// CHECK: %int3 = torch.constant.int 3
%int3 = torch.constant.int 3
// CHECK: %int-3 = torch.constant.int -3
%int-3 = torch.constant.int -3

// CHECK: %float1.000000e00 = torch.constant.float 1.000000e+00
%float1.000000e00 = torch.constant.float 1.000000e+00
// CHECK: %float-1.000000e00 = torch.constant.float -1.000000e+00
%float-1.000000e00 = torch.constant.float -1.000000e+00
// CHECK: %float1.000000e-10 = torch.constant.float 1.000000e-10
%float1.000000e-10 = torch.constant.float 1.000000e-10
// CHECK: %float1.000000e10 = torch.constant.float 1.000000e+10
%float1.000000e10 = torch.constant.float 1.000000e+10
// CHECK: %float4.250000e01 = torch.constant.float 4.250000e+01
%float4.250000e01 = torch.constant.float 4.250000e+01

%tensor = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
// CHECK: %none = torch.constant.none
%none = torch.constant.none
// CHECK: %str = torch.constant.str "some str"
%str = torch.constant.str "some str"
func.func private @f(%arg0: !torch.nn.Module<"test">) {
  return
}

torch.class_type @empty {}
%submodule = torch.nn_module {} : !torch.nn.Module<"empty">

torch.class_type @test {
  torch.attr "b" : !torch.bool
  torch.attr "i" : !torch.int
  torch.attr "f" : !torch.float
  torch.attr "t" : !torch.tensor
  torch.attr "submodule" : !torch.nn.Module<"empty">
  torch.attr "ob" : !torch.optional<bool>
  torch.attr "s" : !torch.str
  torch.method "method", @f
}
torch.nn_module {
  torch.slot "b", %true : !torch.bool
  torch.slot "i", %int3 : !torch.int
  torch.slot "f", %float1.000000e00 : !torch.float
  torch.slot "t", %tensor : !torch.tensor
  torch.slot "submodule", %submodule : !torch.nn.Module<"empty">
  torch.slot "ob", %none : !torch.none
  torch.slot "s", %str : !torch.str
} : !torch.nn.Module<"test">


func.func @shape_calculations(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.shape.calculate {
    %0 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
    torch.shape.calculate.yield %0 : !torch.vtensor
  } shapes {
    %0 = torch.aten.size %arg0 : !torch.vtensor -> !torch.list<int>
    torch.shape.calculate.yield.shapes %0 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

func.func @dtype_calculations(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.dtype.calculate {
    %1 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
    torch.dtype.calculate.yield %1 : !torch.vtensor
  } dtypes {
    %2 = torch.prim.dtype %arg0 : !torch.vtensor -> !torch.int
    torch.dtype.calculate.yield.dtypes %2 : !torch.int
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

func.func @promote_dtypes(%ranks: !torch.list<optional<int>>, %dtypes: !torch.list<int>) -> !torch.int {
  %0 = torch.promote_dtypes %ranks, %dtypes : (!torch.list<optional<int>>, !torch.list<int>) -> !torch.int
  return %0 : !torch.int
}

func.func @number_type_subtypes(%arg0: !torch.tensor, %arg1: !torch.list<int>, %arg2: !torch.union<float, int>) {
  %0 = torch.aten.constant_pad_nd %arg0, %arg1, %arg2 : !torch.tensor, !torch.list<int>, !torch.union<float, int> -> !torch.tensor
  return
}

func.func private @tensor_legal_dtype$torch.qint8() -> !torch.tensor<*,!torch.qint8>
func.func private @tensor_legal_dtype$torch.quint8() -> !torch.tensor<*,!torch.quint8>

func.func @prim_list_construct$valid_shape_subtype(%arg0: !torch.vtensor<[1,53,56,96],f16>, %arg1: !torch.vtensor<[1,3,56,96],f16>) -> !torch.list<vtensor<[1,?,56,96],f16>> {
  %arg2 = "torch.prim.ListConstruct"(%arg0, %arg1) : (!torch.vtensor<[1,53,56,96],f16>, !torch.vtensor<[1,3,56,96],f16>) -> !torch.list<vtensor<[1,?,56,96],f16>>
  return %arg2 : !torch.list<vtensor<[1,?,56,96],f16>>
}

// Check that verification passes with '-1' as a permutation index.
func.func @torch.permute$negative_index_valid (%arg0: !torch.vtensor<[1,2,3],f32>) -> !torch.vtensor<[1,2,3],f32> {
  %intm1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %perm = torch.prim.ListConstruct %int0, %int1, %intm1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.permute %arg0, %perm : !torch.vtensor<[1,2,3],f32>, !torch.list<int> -> !torch.vtensor<[1,2,3],f32>
   return %3 : !torch.vtensor<[1,2,3],f32>
}

