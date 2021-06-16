// RUN: npcomp-opt %s | npcomp-opt | FileCheck %s

// CHECK-LABEL: func @torch.operator(
func @torch.operator(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor {
  // CHECK: torch.operator "ns.unqual.overload"(%arg0, %arg1) : (!torch.tensor, !torch.tensor) -> !torch.tensor
  %0 = torch.operator "ns.unqual.overload"(%arg0, %arg1) : (!torch.tensor, !torch.tensor) -> !torch.tensor
  return %0 : !torch.tensor
}

func @torch.linear_params.create(%arg0: !torch.tensor, %arg1: !torch.tensor) -> (!torch.LinearParams, !torch.LinearParams) {
  %with_bias = torch.linear_params.create %arg0, %arg1 : !torch.tensor, !torch.tensor
  %without_bias = torch.linear_params.create %arg0 : !torch.tensor
  return %with_bias, %without_bias : !torch.LinearParams, !torch.LinearParams
}

// CHECK-LABEL: func @builtin_tensor_interop(
func @builtin_tensor_interop(%arg0: tensor<*xf32>, %arg1: tensor<3x?xsi8>, %arg2: !torch.vtensor<*,f32>, %arg3: !torch.vtensor<[3,?],si8>) {
  // CHECK: torch.from_builtin_tensor %arg0 : tensor<*xf32> -> !torch.vtensor<*,f32>
  %0 = torch.from_builtin_tensor %arg0 : tensor<*xf32> -> !torch.vtensor<*,f32>
  // CHECK: torch.from_builtin_tensor %arg1 : tensor<3x?xsi8> -> !torch.vtensor<[3,?],si8>
  %1 = torch.from_builtin_tensor %arg1 : tensor<3x?xsi8> -> !torch.vtensor<[3,?],si8>
  // CHECK: torch.to_builtin_tensor %arg2 : !torch.vtensor<*,f32> -> tensor<*xf32>
  %2 = torch.to_builtin_tensor %arg2 : !torch.vtensor<*,f32> -> tensor<*xf32>
  // CHECK: torch.to_builtin_tensor %arg3 : !torch.vtensor<[3,?],si8> -> tensor<3x?xsi8>
  %3 = torch.to_builtin_tensor %arg3 : !torch.vtensor<[3,?],si8> -> tensor<3x?xsi8>
  return
}

// CHECK: @tensor.default() -> !torch.tensor
func private @tensor.default() -> !torch.tensor
// CHECK: @tensor.default_explicit() -> !torch.tensor{{$}}
func private @tensor.default_explicit() -> !torch.tensor<*,unk>
// CHECK: @tensor.value_semantic() -> !torch.vtensor{{$}}
func private @tensor.value_semantic() -> !torch.vtensor<*,unk>
// CHECK: @tensor.dtype() -> !torch.tensor<*,si32>
func private @tensor.dtype() -> !torch.tensor<*,si32>
// CHECK: @tensor.ranked() -> !torch.tensor<[?,?,?],unk>
func private @tensor.ranked() -> !torch.tensor<[?,?,?],unk>
// CHECK: @tensor.some_sizes_known() -> !torch.tensor<[?,2,?,4],unk>
func private @tensor.some_sizes_known() -> !torch.tensor<[?,2,?,4],unk>
// CHECK: @tensor.fully_determined() -> !torch.vtensor<[1,2,3,4],f32>
func private @tensor.fully_determined() -> !torch.vtensor<[1,2,3,4],f32>

// CHECK: @tuple.empty() -> !torch.tuple<>
func private @tuple.empty() -> !torch.tuple<>
// CHECK: @tuple.one_element() -> !torch.tuple<!torch.tensor>
func private @tuple.one_element() -> !torch.tuple<!torch.tensor>
// CHECK: @tuple.two_elements() -> !torch.tuple<!torch.tensor, !torch.tensor>
func private @tuple.two_elements() -> !torch.tuple<!torch.tensor, !torch.tensor>

// CHECK-LABEL:   func @torch.tensor() {
func @torch.tensor() {
  // CHECK: torch.tensor(dense<4.200000e+01> : tensor<3x2xf32>) : !torch.vtensor<[3,2],f32>
  %0 = torch.tensor(dense<42.0> : tensor<3x2xf32>) : !torch.vtensor<[3,2],f32>
  // CHECK: torch.tensor(dense<4.200000e+01> : tensor<3x2xf32>) : !torch.tensor<[3,2],f32>
  %1 = torch.tensor(dense<42.0> : tensor<3x2xf32>) : !torch.tensor<[3,2],f32>
  return
}

func @derefine(%arg0: !torch.tensor) -> !torch.optional<!torch.tensor> {
  %0 = torch.derefine %arg0 : !torch.tensor to !torch.optional<!torch.tensor>
  return %0 : !torch.optional<!torch.tensor>
}

func @torch.prim.If(%arg0: !torch.bool, %arg1: !torch.int) -> !torch.int {
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
// CHECK: %float = torch.constant.float 4.250000e+01
%float = torch.constant.float 4.250000e+01
%tensor = torch.tensor(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor
// CHECK: %none = torch.constant.none
%none = torch.constant.none
// CHECK: %str = torch.constant.str "some str"
%str = torch.constant.str "some str"
func private @f(%arg0: !torch.nn.Module<"test">) {
  return
}

torch.class_type @empty {}
%submodule = torch.nn_module {} : !torch.nn.Module<"empty">

torch.class_type @test {
  torch.attr "b" : !torch.bool
  torch.attr "i" : !torch.int
  torch.attr "f" : f64
  torch.attr "t" : !torch.tensor
  torch.attr "submodule" : !torch.nn.Module<"empty">
  torch.attr "ob" : !torch.optional<!torch.bool>
  torch.attr "s" : !torch.str
  torch.method "method", @f
}
torch.nn_module {
  torch.slot "b", %true : !torch.bool
  torch.slot "i", %int3 : !torch.int
  torch.slot "f", %float : f64
  torch.slot "t", %tensor : !torch.tensor
  torch.slot "submodule", %submodule : !torch.nn.Module<"empty">
  torch.slot "ob", %none : !torch.none
  torch.slot "s", %str : !torch.str
} : !torch.nn.Module<"test">
