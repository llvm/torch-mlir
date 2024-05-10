// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// Since we want to migrate to the strict view op lowering, these test cases
// verify this one pattern specifically via attributes on the functions that
// disable the legacy behavior.

// -----

// CHECK-LABEL:   func.func @torch.aten.view$twotothree
// CHECK:       %[[ARG0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[3,2],f32> -> tensor<3x2xf32>
// CHECK:       %[[T3:.*]] = torch.constant.int 3
// CHECK:       %[[T2:.*]] = torch.constant.int 2
// CHECK:       %[[N2:.*]] = torch_c.to_i64 %[[T2]]
// CHECK:       %[[N3:.*]] = torch_c.to_i64 %[[T3]]
// CHECK:       %[[ELEMENTS:.*]] = tensor.from_elements %[[N2]], %[[N3]] : tensor<2xi64>
// CHECK:       %[[RESHAPE:.*]] = tensor.reshape %[[ARG0]](%[[ELEMENTS]]) : (tensor<3x2xf32>, tensor<2xi64>) -> tensor<2x3xf32>
func.func @torch.aten.view$twotothree(%arg0: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,3],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[3,2],f32>, !torch.list<int> -> !torch.vtensor<[2,3],f32>
    return %1 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.view$zerod
// CHECK:       %[[ARG0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK:       tensor.collapse_shape %0 [] : tensor<?x?xf32> into tensor<f32>
func.func @torch.aten.view$zerod(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
    %0 = torch.prim.ListConstruct : () -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
    return %1 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$dynamictest
// CHECK:       %[[ARG0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK:       %[[ARG1:.*]] = torch_c.to_i64 %arg1
// CHECK:       %[[ARG2:.*]] = torch_c.to_i64 %arg2
// CHECK:       %[[ELTS:.*]] = tensor.from_elements %[[ARG1]], %[[ARG2]] : tensor<2xi64>
// CHECK:       tensor.reshape %[[ARG0]](%[[ELTS]]) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
func.func @torch.aten.view$dynamictest(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[?,?],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
    %2 = torch.prim.ListConstruct %arg1, %arg2 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
    return %3 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$dynamictest2(
// CHECK-SAME:      %[[ARG:.*]]: !torch.vtensor<[?,6,?],f32>) -> !torch.vtensor<[?,2,3,?],f32>
// CHECK:        %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,6,?],f32> -> tensor<?x6x?xf32>
// CHECK:        %[[EXPAND:.*]] = tensor.reshape %[[BUILTIN_TENSOR]]
// CHECK:        %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND]] : tensor<?x2x3x?xf32> -> !torch.vtensor<[?,2,3,?],f32>
// CHECK:        return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[?,2,3,?],f32>

func.func @torch.aten.view$dynamictest2(%arg0: !torch.vtensor<[?,6,?],f32>) -> !torch.vtensor<[?,2,3,?],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %2 = torch.aten.size.int %arg0, %int2 : !torch.vtensor<[?,6,?],f32>, !torch.int -> !torch.int
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,6,?],f32>, !torch.int -> !torch.int
  %1 = torch.prim.ListConstruct %0, %int2, %int3, %2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.view %arg0, %1 : !torch.vtensor<[?,6,?],f32>, !torch.list<int> -> !torch.vtensor<[?,2,3,?], f32>
  return %3 : !torch.vtensor<[?,2,3,?], f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$dynamicVal(
// CHECK:       tensor.reshape {{.*}} : (tensor<1x?x128xf32>, tensor<3xi64>) -> tensor<16x1x128xf32>
func.func @torch.aten.view$dynamicVal(%arg0: !torch.vtensor<[1,?,128],f32>) -> !torch.vtensor<[16,1,128],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
    %int128 = torch.constant.int 128
    %int1 = torch.constant.int 1
    %int16 = torch.constant.int 16
    %0 = torch.prim.ListConstruct %int16, %int1, %int128 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1,?,128],f32>, !torch.list<int> -> !torch.vtensor<[16,1,128],f32>
    return %1 : !torch.vtensor<[16,1,128],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$expandInferredDim
// CHECK:       %[[ARG0:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK:       %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1]] : tensor<2x6xf32> into tensor<12xf32>
// CHECK:       %[[CAST1:.*]] = tensor.cast %[[COLLAPSED]] : tensor<12xf32> to tensor<12xf32>
// CHECK:       %[[EXPANDED:.*]] = tensor.expand_shape %[[CAST1]] {{\[\[}}0, 1, 2]] output_shape [3, 2, 2] : tensor<12xf32> into tensor<3x2x2xf32>
func.func @torch.aten.view$expandInferredDim(%arg0: !torch.vtensor<[2,6],f32>) -> !torch.vtensor<[3,2,2],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int3, %int2, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,6],f32>, !torch.list<int> -> !torch.vtensor<[3,2,2],f32>
    return %1 : !torch.vtensor<[3,2,2],f32>
}

// -----
// TODO: Enable once supported.
func.func @torch.aten$dynamicValOutput(%arg0: !torch.vtensor<[?, ?, ?],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,1,?,1],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %arg1, %int1, %int-1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // expected-error@+1 {{torch.aten.view}}
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?, ?, ?],f32>, !torch.list<int> -> !torch.vtensor<[?,1,?,1],f32>
  return %1 : !torch.vtensor<[?,1,?,1],f32>
}
