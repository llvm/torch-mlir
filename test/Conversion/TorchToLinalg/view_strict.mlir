// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// Since we want to migrate to the strict view op lowering, these test cases
// verify this one pattern specifically via attributes on the functions that
// disable the legacy behavior.

// -----

// CHECK-LABEL:   func.func @torch.aten.view$twotothree
// CHECK:       %[[ARG0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[3,2],f32> -> tensor<3x2xf32>
// CHECK:       %[[N2:.*]] = arith.constant 2 : i64
// CHECK:       %[[N3:.*]] = arith.constant 3 : i64
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
// Note that this is presently going down a fallback path as an explicit
// reshape. Someday, this should generate flatten/unflatten.
// CHECK-LABEL: func.func @torch.aten$dynamicValOutput
// CHECK:       %[[SELF:.*]] = torch_c.to_builtin_tensor %arg0
// CHECK-DAG:   %[[PROD1:.*]] = arith.constant 1
// CHECK-DAG:   %[[ARG1_CVT:.*]] = torch_c.to_i64 %arg1
// CHECK-DAG:   %[[PROD2:.*]] = arith.muli %[[PROD1]], %[[ARG1_CVT]]
// CHECK-DAG:   %[[ONEI64:.*]] = arith.constant 1 : i64
// CHECK-DAG:   %[[PROD3:.*]] = arith.muli %[[PROD2]], %[[ONEI64]]
// CHECK-DAG:   %[[ONEI64_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:   %[[PROD4:.*]] = arith.muli %[[PROD3]], %[[ONEI64_0]]
// CHECK-DAG:   %[[INDEX0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[DIM0_INDEX:.*]] = tensor.dim %[[SELF]], %[[INDEX0]] : tensor<?x?x?xf32>
// CHECK-DAG:   %[[DIM0:.*]] = arith.index_cast %[[DIM0_INDEX]] : index to i64
// CHECK-DAG:   %[[KNOWN0:.*]] = arith.muli %[[PROD1]], %[[DIM0]] : i64
// CHECK-DAG:   %[[INDEX1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[DIM1_INDEX:.*]] = tensor.dim %[[SELF]], %[[INDEX1]] : tensor<?x?x?xf32>
// CHECK-DAG:   %[[DIM1:.*]] = arith.index_cast %[[DIM1_INDEX]] : index to i64
// CHECK-DAG:   %[[KNOWN1:.*]] = arith.muli %[[KNOWN0]], %[[DIM1]] : i64
// CHECK-DAG:   %[[INDEX2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[DIM2_INDEX:.*]] = tensor.dim %[[SELF]], %[[INDEX2]] : tensor<?x?x?xf32>
// CHECK-DAG:   %[[DIM2:.*]] = arith.index_cast %[[DIM2_INDEX]] : index to i64
// CHECK-DAG:   %[[KNOWN2:.*]] = arith.muli %[[KNOWN1]], %[[DIM2]] : i64
// CHECK-DAG:   %[[DIMINFER:.*]] = arith.divui %[[KNOWN2]], %[[PROD4]] : i64
// CHECK:       %[[DIM0:.*]] = torch_c.to_i64 %arg1
// CHECK:       %[[DIM1:.*]] = arith.constant 1 : i64
// CHECK:       %[[DIM3:.*]] = arith.constant 1 : i64
// CHECK:       %[[OUTPUT_DIMS:.*]] = tensor.from_elements %[[DIM0]], %[[DIM1]], %[[DIMINFER]], %[[DIM3]] : tensor<4xi64>
// CHECK:       tensor.reshape %[[SELF]](%[[OUTPUT_DIMS]]) : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x1x?x1xf32>
//
func.func @torch.aten$dynamicValOutput(%arg0: !torch.vtensor<[?, ?, ?],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,1,?,1],f32>
  attributes {torch.assume_strict_symbolic_shapes, torch.disable_legacy_view}
{
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %arg1, %int1, %int-1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?, ?, ?],f32>, !torch.list<int> -> !torch.vtensor<[?,1,?,1],f32>
  return %1 : !torch.vtensor<[?,1,?,1],f32>
}
