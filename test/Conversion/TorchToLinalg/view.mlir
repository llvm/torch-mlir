// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.view$twotothree(
// CHECK-SAME:  %[[ARG:.*]]: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,3],f32> {
// CHECK:    %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[3,2],f32> -> tensor<3x2xf32>
// CHECK:    %[[CASTED:.*]] = tensor.cast %[[BUILTIN_TENSOR]] : tensor<3x2xf32> to tensor<3x2xf32>
// CHECK:    %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CASTED]] {{\[\[}}0, 1]] : tensor<3x2xf32> into tensor<6xf32>
// CHECK:    %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1]] : tensor<6xf32> into tensor<2x3xf32>
// CHECK:    %[[EXPAND_CAST:.*]] = tensor.cast %[[EXPANDED]] : tensor<2x3xf32> to tensor<2x3xf32>
// CHECK:    %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND_CAST]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:    return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[2,3],f32>

func.func @torch.aten.view$twotothree(%arg0: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,3],f32> {
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[3,2],f32>, !torch.list<int> -> !torch.vtensor<[2,3],f32>
    return %1 : !torch.vtensor<[2,3],f32>
  }

// CHECK-LABEL: func.func @torch.aten.view$dynamictest(
// CHECK-SAME:      %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:        %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[CASTED:.*]] = tensor.cast %[[BUILTIN_TENSOR]] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:        %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[CASTED]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
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

// CHECK-LABEL: func.func @torch.aten.view$dynamicVal(
// CHECK-SAME:     %[[ARG:.*]]: !torch.vtensor<[1,?,128],f32>) -> !torch.vtensor<[16,1,128],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[1,?,128],f32> -> tensor<1x?x128xf32>
// CHECK:     %[[CASTED:.*]] = tensor.cast %[[BUILTIN_TENSOR]] : tensor<1x?x128xf32> to tensor<1x16x128xf32>
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CASTED]] {{\[\[}}0, 1], [2]] : tensor<1x16x128xf32> into tensor<16x128xf32>
// CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0], [1, 2]] : tensor<16x128xf32> into tensor<16x1x128xf32>
// CHECK:     %[[EXPAND_CAST:.*]] = tensor.cast %[[EXPANDED]] : tensor<16x1x128xf32> to tensor<16x1x128xf32>
// CHECK:     %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND_CAST]] : tensor<16x1x128xf32> -> !torch.vtensor<[16,1,128],f32>
// CHECK:     return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[16,1,128],f32>

func.func @torch.aten.view$dynamicVal(%arg0: !torch.vtensor<[1,?,128],f32>) -> !torch.vtensor<[16,1,128],f32> {
    %int128 = torch.constant.int 128
    %int1 = torch.constant.int 1
    %int16 = torch.constant.int 16
    %0 = torch.prim.ListConstruct %int16, %int1, %int128 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1,?,128],f32>, !torch.list<int> -> !torch.vtensor<[16,1,128],f32>
    return %1 : !torch.vtensor<[16,1,128],f32>
  }

// CHECK-LABEL: func.func @torch.aten.view$expandInferredDim(
// CHECK-SAME:    %[[ARG:.*]]: !torch.vtensor<[2,6],f32>) -> !torch.vtensor<[3,2,2],f32> {
// CHECK:     %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[2,6],f32> -> tensor<2x6xf32>
// CHECK:     %[[CASTED:.*]] = tensor.cast %[[BUILTIN_TENSOR]] : tensor<2x6xf32> to tensor<2x6xf32>
// CHECK:     %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CASTED]] {{\[\[}}0, 1]] : tensor<2x6xf32> into tensor<12xf32>
// CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[COLLAPSED]] {{\[\[}}0, 1, 2]] : tensor<12xf32> into tensor<3x2x2xf32>
// CHECK:     %[[EXPAND_CAST:.*]] = tensor.cast %[[EXPANDED]] : tensor<3x2x2xf32> to tensor<3x2x2xf32>
// CHECK:     %[[BUILTIN_TENSOR_CAST:.*]] = torch_c.from_builtin_tensor %[[EXPAND_CAST]] : tensor<3x2x2xf32> -> !torch.vtensor<[3,2,2],f32>
// CHECK:     return %[[BUILTIN_TENSOR_CAST]] : !torch.vtensor<[3,2,2],f32>

func.func @torch.aten.view$expandInferredDim(%arg0: !torch.vtensor<[2,6],f32>) -> !torch.vtensor<[3,2,2],f32> {
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int3, %int2, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[2,6],f32>, !torch.list<int> -> !torch.vtensor<[3,2,2],f32>
    return %1 : !torch.vtensor<[3,2,2],f32>
  }