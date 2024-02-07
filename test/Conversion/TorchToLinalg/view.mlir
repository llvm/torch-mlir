// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.view$twotothree(
// CHECK-SAME:  %[[ARG:.*]]: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:    %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[3,2],f32> -> tensor<3x2xf32>
// CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1]] : tensor<3x2xf32> into tensor<6xf32>
// CHECK:    %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1]] : tensor<6xf32> into tensor<2x3xf32>
// CHECK:    %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:    return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[2,3],f32>

func.func @torch.aten.view$twotothree(%arg0: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,3],f32> {
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[3,2],f32>, !torch.list<int> -> !torch.vtensor<[2,3],f32>
    return %1 : !torch.vtensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$dynamictest(
// CHECK-SAME:      %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:        %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[BUILTIN_TENSOR]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:        return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[?,?],f32>

func.func @torch.aten.view$dynamictest(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    %2 = torch.prim.ListConstruct %0, %1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
    return %3 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$dynamictest2(
// CHECK-SAME:      %[[ARG:.*]]: !torch.vtensor<[?,6,?],f32>) -> !torch.vtensor<[?,2,3,?],f32> {
// CHECK:        %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,6,?],f32> -> tensor<?x6x?xf32>
// CHECK:        %[[EXPAND:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1, 2], [3]] : tensor<?x6x?xf32> into tensor<?x2x3x?xf32>
// CHECK:        %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND]] : tensor<?x2x3x?xf32> -> !torch.vtensor<[?,2,3,?],f32>
// CHECK:        return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[?,2,3,?],f32>

func.func @torch.aten.view$dynamictest2(%arg0: !torch.vtensor<[?,6,?],f32>) -> !torch.vtensor<[?,2,3,?],f32> {
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
// CHECK-SAME:     %[[ARG:.*]]: !torch.vtensor<[1,?,128],f32>) -> !torch.vtensor<[16,1,128],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[1,?,128],f32> -> tensor<1x?x128xf32>
// CHECK:     %[[CASTED:.*]] = tensor.cast %[[BUILTIN_TENSOR]] : tensor<1x?x128xf32> to tensor<1x16x128xf32>
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CASTED]] {{\[\[}}0, 1], [2]] : tensor<1x16x128xf32> into tensor<16x128xf32>
// CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0], [1, 2]] : tensor<16x128xf32> into tensor<16x1x128xf32>
// CHECK:     %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<16x1x128xf32> -> !torch.vtensor<[16,1,128],f32>
// CHECK:     return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[16,1,128],f32>

func.func @torch.aten.view$dynamicVal(%arg0: !torch.vtensor<[1,?,128],f32>) -> !torch.vtensor<[16,1,128],f32> {
    %int128 = torch.constant.int 128
    %int1 = torch.constant.int 1
    %int16 = torch.constant.int 16
    %0 = torch.prim.ListConstruct %int16, %int1, %int128 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1,?,128],f32>, !torch.list<int> -> !torch.vtensor<[16,1,128],f32>
    return %1 : !torch.vtensor<[16,1,128],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten$dynamicValOutput(
// CHECK-SAME:     %[[ARG:.*]]: !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[8,1,?,1],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[4,5,6],f32> -> tensor<4x5x6xf32>
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1, 2]] : tensor<4x5x6xf32> into tensor<120xf32>
// CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2, 3]] : tensor<120xf32> into tensor<8x1x15x1xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[EXPANDED]] : tensor<8x1x15x1xf32> to tensor<8x1x?x1xf32>
// CHECK: %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<8x1x?x1xf32> -> !torch.vtensor<[8,1,?,1],f32>
// CHECK: return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[8,1,?,1],f32>

func.func @torch.aten$dynamicValOutput(%arg0: !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[8,1,?,1],f32> {
  %int8 = torch.constant.int 8
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int8, %int1, %int-1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[4,5,6],f32>, !torch.list<int> -> !torch.vtensor<[8,1,?,1],f32>
  return %1 : !torch.vtensor<[8,1,?,1],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten$dynamicValOutput2(
// CHECK-SAME:     %[[ARG:.*]]: !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[2,1,2,3,?],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[4,5,6],f32> -> tensor<4x5x6xf32>
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1, 2]] : tensor<4x5x6xf32> into tensor<4x30xf32>
// CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2], [3, 4]] : tensor<4x30xf32> into tensor<2x1x2x3x10xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[EXPANDED]] : tensor<2x1x2x3x10xf32> to tensor<2x1x2x3x?xf32>
// CHECK: %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<2x1x2x3x?xf32> -> !torch.vtensor<[2,1,2,3,?],f32>
// CHECK: return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[2,1,2,3,?],f32>

// 4 -> [2,1,2] [5,6] -> [3,10].
func.func @torch.aten$dynamicValOutput2(%arg0: !torch.vtensor<[4,5,6],f32>) -> !torch.vtensor<[2,1,2,3,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int2, %int1, %int2, %int3, %int-1 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[4,5,6],f32>, !torch.list<int> -> !torch.vtensor<[2,1,2,3,?],f32>
  return %1 : !torch.vtensor<[2,1,2,3,?],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$expandInferredDim(
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[2,6],f32>) -> !torch.vtensor<[3,2,2],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[2,6],f32> -> tensor<2x6xf32>
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1]] : tensor<2x6xf32> into tensor<12xf32>
// CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2]] : tensor<12xf32> into tensor<3x2x2xf32>
// CHECK:     %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<3x2x2xf32> -> !torch.vtensor<[3,2,2],f32>
// CHECK:     return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[3,2,2],f32>

func.func @torch.aten.view$expandInferredDim(%arg0: !torch.vtensor<[2,6],f32>) -> !torch.vtensor<[3,2,2],f32> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int3, %int2, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,6],f32>, !torch.list<int> -> !torch.vtensor<[3,2,2],f32>
    return %1 : !torch.vtensor<[3,2,2],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$singleUnknownMatches0(
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[10,3,?,2,3],f32>) -> !torch.vtensor<[2,3,5,?,6],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[10,3,?,2,3],f32> -> tensor<10x3x?x2x3xf32>
// CHECK:   %[[COLLAPSE:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1], [2], [3, 4]] : tensor<10x3x?x2x3xf32> into tensor<30x?x6xf32>
// CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[COLLAPSE]] {{\[\[}}0, 1, 2], [3], [4]] : tensor<30x?x6xf32> into tensor<2x3x5x?x6xf32>
// CHECK: %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND]] : tensor<2x3x5x?x6xf32> -> !torch.vtensor<[2,3,5,?,6],f32>
// CHECK: return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[2,3,5,?,6],f32>

// [10,3,?,2,3] -> [30,?,6] -> [2,3,5,?,6]
// Associations are,
//  -- for collapse, [0,1], [2], [3,4] and
//  -- for expand [0,1,2], [3], [4].
func.func @torch.aten.view$singleUnknownMatches0(%arg0: !torch.vtensor<[10,3,?,2,3],f32>) -> !torch.vtensor<[2,3,5,?,6],f32> {
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int6 = torch.constant.int 6
    %int5 = torch.constant.int 5
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int2, %int3, %int5, %int-1, %int6 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[10,3,?,2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,3,5,?,6],f32>
    return %1 : !torch.vtensor<[2,3,5,?,6],f32>
}

// -----

// Multiple aspects of decomposition here:
// 1) an expand from (8) to (2,2,2)
// 2) a collapse from (2,1,3) to (6)
// 3) a single unknown dim matching in the middle.
// 4) on either side of the unkown dim (3), another unkown dim,
// but one which matches between the input and the output

// CHECK: func.func @torch.aten.view$combineConcepts(
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[8,?,?,?,2,1,3],f32>) -> !torch.vtensor<[2,2,2,?,?,?,6],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[8,?,?,?,2,1,3],f32> -> tensor<8x?x?x?x2x1x3xf32>
// CHECK:   %[[COLLAPSE:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1], [2], [3], [4, 5, 6]] : tensor<8x?x?x?x2x1x3xf32> into tensor<8x?x?x?x6xf32>
// CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[COLLAPSE]] {{\[\[}}0, 1, 2], [3], [4], [5], [6]] : tensor<8x?x?x?x6xf32> into tensor<2x2x2x?x?x?x6xf32>
// CHECK: %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND]] : tensor<2x2x2x?x?x?x6xf32> -> !torch.vtensor<[2,2,2,?,?,?,6],f32>
// CHECK: return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[2,2,2,?,?,?,6],f32>

func.func @torch.aten.view$combineConcepts(%arg0 : !torch.vtensor<[8,?,?,?,2,1,3], f32>) -> !torch.vtensor<[2,2,2,?,?,?,6], f32> {

  %int1 = torch.constant.int 1
  %size1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[8,?,?,?,2,1,3], f32>, !torch.int -> !torch.int

  %int3 = torch.constant.int 3
  %size3 = torch.aten.size.int %arg0, %int3 : !torch.vtensor<[8,?,?,?,2,1,3], f32>, !torch.int -> !torch.int

  %int2 = torch.constant.int 2
  %int6 = torch.constant.int 6
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int2, %int2, %int2, %size1, %int-1, %size3, %int6 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[8,?,?,?,2,1,3], f32>, !torch.list<int> -> !torch.vtensor<[2,2,2,?,?,?,6], f32>
  return %1 : !torch.vtensor<[2,2,2,?,?,?,6], f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$multiDynamicsInSourceOfCollapse
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[?,2,?,4,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,2,?,4,?],f32> -> tensor<?x2x?x4x?xf32>
// CHECK:   %[[COLLAPSE:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1, 2, 3, 4]] : tensor<?x2x?x4x?xf32> into tensor<?xf32>
// CHECK: %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[COLLAPSE]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK: return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[?],f32>
func.func @torch.aten.view$multiDynamicsInSourceOfCollapse (%arg0 : !torch.vtensor<[?,2,?,4,?], f32>) -> !torch.vtensor<[?], f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,2,?,4,?], f32>, !torch.list<int> -> !torch.vtensor<[?], f32>
  return %1 : !torch.vtensor<[?], f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view$castingView
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[3,4,5],f32> {

// The current lowring only succeeds if the input (arg0) has shape [3,4,5],
// determined at runtime. This is a bit limiting, and we'll probably want to
// improve that in the future. For now we check that there are 2 runtime
// asserts on the sizes of dimensions 0 and 1 (size of dimension 2 implied).

// CHECK-COUNT-2:    cf.assert {{.*}} "mismatching contracting dimension
// CHECK: return {{.*}} : !torch.vtensor<[3,4,5],f32>

func.func @torch.aten.view$castingView (%arg0 : !torch.vtensor<[?,?,?], f32>) -> !torch.vtensor<[3,4,5], f32> {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %0 = torch.prim.ListConstruct %int3, %int4, %int5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[?,?,?], f32>, !torch.list<int> -> !torch.vtensor<[3,4,5], f32>
  return %1 : !torch.vtensor<[3,4,5], f32>
}

// -----

// A function with a torch.view op, going from shape (10,?,2,3) to (2,5,?,6).
// We expect this to lower to a collapse with [0], [1], [2,3] followed by
// an expand with [0,1], [2], [3]:
// CHECK: func.func @torch.aten.view$dynamicInferredSame(
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[10,?,2,3],f32>) -> !torch.vtensor<[2,5,?,6],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[10,?,2,3],f32> -> tensor<10x?x2x3xf32>
// CHECK:   %[[COLLAPSE:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1], [2, 3]] : tensor<10x?x2x3xf32> into tensor<10x?x6xf32>
// CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[COLLAPSE]] {{\[\[}}0, 1], [2], [3]] : tensor<10x?x6xf32> into tensor<2x5x?x6xf32>
// CHECK: %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND]] : tensor<2x5x?x6xf32> -> !torch.vtensor<[2,5,?,6],f32>
// CHECK: return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[2,5,?,6],f32>

func.func @torch.aten.view$dynamicInferredSame(%arg0: !torch.vtensor<[10,?,2,3],f32>) -> !torch.vtensor<[2,5,?,6],f32> {
  %int2 = torch.constant.int 2
  %int5 = torch.constant.int 5
  %int6 = torch.constant.int 6
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int2, %int5, %int-1, %int6 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[10,?,2,3],f32>, !torch.list<int> -> !torch.vtensor<[2,5,?,6],f32>
  return %1 : !torch.vtensor<[2,5,?,6],f32>
}

