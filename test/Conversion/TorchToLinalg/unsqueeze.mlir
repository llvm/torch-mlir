// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s


// -----

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$basic(
// CHECK-SAME:                                     %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @torch.aten.unsqueeze$basic(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$basic_negative(
// CHECK-SAME:                                              %[[ARG:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @torch.aten.unsqueeze$basic_negative(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int-1 : !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$higher_rank_front(
// CHECK-SAME:                                                 %[[ARG:.*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,2,3,4],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1], [2], [3]]  output_shape [1, 2, 3, 4] : tensor<2x3x4xf32> into tensor<1x2x3x4xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<1x2x3x4xf32> -> !torch.vtensor<[1,2,3,4],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1,2,3,4],f32>
func.func @torch.aten.unsqueeze$higher_rank_front(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,2,3,4],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[2,3,4],f32>, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  return %0 : !torch.vtensor<[1,2,3,4],f32>
}

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$higher_rank_back(
// CHECK-SAME:                                                %[[ARG:.*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4,1],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1], [2, 3]] output_shape [2, 3, 4, 1] : tensor<2x3x4xf32> into tensor<2x3x4x1xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<2x3x4x1xf32> -> !torch.vtensor<[2,3,4,1],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[2,3,4,1],f32>
func.func @torch.aten.unsqueeze$higher_rank_back(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4,1],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int-1 : !torch.vtensor<[2,3,4],f32>, !torch.int -> !torch.vtensor<[2,3,4,1],f32>
  return %0 : !torch.vtensor<[2,3,4,1],f32>
}

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$higher_rank_middle(
// CHECK-SAME:                                                  %[[ARG:.*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,1,4],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1], [2, 3]] output_shape [2, 3, 1, 4] : tensor<2x3x4xf32> into tensor<2x3x1x4xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[EXPANDED]] : tensor<2x3x1x4xf32> -> !torch.vtensor<[2,3,1,4],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[2,3,1,4],f32>
func.func @torch.aten.unsqueeze$higher_rank_middle(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,1,4],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[2,3,4],f32>, !torch.int -> !torch.vtensor<[2,3,1,4],f32>
  return %0 : !torch.vtensor<[2,3,1,4],f32>
}
