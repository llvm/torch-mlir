// RUN: npcomp-opt %s | npcomp-opt | FileCheck %s

// CHECK-LABEL: func @torch.operator(
func @torch.operator(%arg0: !numpy.ndarray<*:!numpy.any_dtype>, %arg1: !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype> {
  // CHECK: torch.operator "ns.unqual.overload"(%arg0, %arg1) : (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype>
  %0 = torch.operator "ns.unqual.overload"(%arg0, %arg1) : (!numpy.ndarray<*:!numpy.any_dtype>, !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype>
  return %0 : !numpy.ndarray<*:!numpy.any_dtype>
}

func @derefine(%arg0: tensor<f32>) -> !torch.optional<tensor<f32>> {
  %0 = torch.derefine %arg0 : tensor<f32> to !torch.optional<tensor<f32>>
  return %0 : !torch.optional<tensor<f32>>
}

%bool_true = basicpy.bool_constant true
%num3_i64 = basicpy.numeric_constant 3 : i64
%num = basicpy.numeric_constant 4.250000e+01 : f64
%cst = constant dense<1.000000e+00> : tensor<1xf32>
%array = numpy.create_array_from_tensor %cst : (tensor<1xf32>) -> !numpy.ndarray<*:!numpy.any_dtype>
%none = basicpy.singleton : !basicpy.NoneType
func private @f(%arg0: !torch.nn.Module<"test">) {
  return
}

torch.class_type @empty {}
%submodule = torch.nn_module {} : !torch.nn.Module<"empty">

torch.class_type @test {
  torch.attr "b" : !basicpy.BoolType
  torch.attr "i" : i64
  torch.attr "f" : f64
  torch.attr "t" : !numpy.ndarray<*:!numpy.any_dtype>
  torch.attr "submodule" : !torch.nn.Module<"empty">
  torch.attr "ob" : !torch.optional<!basicpy.BoolType>
  torch.method "method", @f
}
torch.nn_module {
  torch.slot "b", %bool_true : !basicpy.BoolType
  torch.slot "i", %num3_i64 : i64
  torch.slot "f", %num : f64
  torch.slot "t", %array : !numpy.ndarray<*:!numpy.any_dtype>
  torch.slot "submodule", %submodule : !torch.nn.Module<"empty">
  torch.slot "ob", %none : !basicpy.NoneType
} : !torch.nn.Module<"test">
