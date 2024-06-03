// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints$basic(
// CHECK-SAME:                                           %[[TENSOR:.*]]: !torch.vtensor<[3,3,2,2,3,3,5],f32>) -> !torch.vtensor<[3,3,?,3,5],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[3,3,2,2,3,3,5],f32> -> tensor<3x3x2x2x3x3x5xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1], [2, 3, 4], [5], [6]] : tensor<3x3x2x2x3x3x5xf32> into tensor<3x3x12x3x5xf32>
// CHECK:           %[[DYNAMIC:.*]] = tensor.cast %[[COLLAPSED]] : tensor<3x3x12x3x5xf32> to tensor<3x3x?x3x5xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[DYNAMIC]] : tensor<3x3x?x3x5xf32> -> !torch.vtensor<[3,3,?,3,5],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[3,3,?,3,5],f32>

func.func @torch.aten.flatten.using_ints$basic(%arg0: !torch.vtensor<[3,3,2,2,3,3,5],f32>) -> !torch.vtensor<[3,3,?,3,5],f32> {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.aten.flatten.using_ints %arg0, %int2, %int4 : !torch.vtensor<[3,3,2,2,3,3,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,3,?,3,5],f32>
  return %0 : !torch.vtensor<[3,3,?,3,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints$basic_negative(
// CHECK-SAME:                                           %[[TENSOR:.*]]: !torch.vtensor<[3,3,2,2,3,3,5],f32>) -> !torch.vtensor<[3,3,?,3,5],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[3,3,2,2,3,3,5],f32> -> tensor<3x3x2x2x3x3x5xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1], [2, 3, 4], [5], [6]] : tensor<3x3x2x2x3x3x5xf32> into tensor<3x3x12x3x5xf32>
// CHECK:           %[[DYNAMIC:.*]] = tensor.cast %[[COLLAPSED]] : tensor<3x3x12x3x5xf32> to tensor<3x3x?x3x5xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[DYNAMIC]] : tensor<3x3x?x3x5xf32> -> !torch.vtensor<[3,3,?,3,5],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[3,3,?,3,5],f32>

func.func @torch.aten.flatten.using_ints$basic_negative(%arg0: !torch.vtensor<[3,3,2,2,3,3,5],f32>) -> !torch.vtensor<[3,3,?,3,5],f32> {
  %int-5 = torch.constant.int -5
  %int-3 = torch.constant.int -3
  %0 = torch.aten.flatten.using_ints %arg0, %int-5, %int-3 : !torch.vtensor<[3,3,2,2,3,3,5],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,3,?,3,5],f32>
  return %0 : !torch.vtensor<[3,3,?,3,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints$flatten_front(
// CHECK-SAME:                                                              %[[TENSOR:.*]]: !torch.vtensor<[3,3,2,2],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[3,3,2,2],f32> -> tensor<3x3x2x2xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1, 2], [3]] : tensor<3x3x2x2xf32> into tensor<18x2xf32>
// CHECK:           %[[DYNAMIC:.*]] = tensor.cast %[[COLLAPSED]] : tensor<18x2xf32> to tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[DYNAMIC]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>

func.func @torch.aten.flatten.using_ints$flatten_front(%arg0: !torch.vtensor<[3,3,2,2],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %0 = torch.aten.flatten.using_ints %arg0, %int0, %int2 : !torch.vtensor<[3,3,2,2],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints$flatten_back(
// CHECK-SAME:                                                              %[[TENSOR:.*]]: !torch.vtensor<[3,3,2,2],f32>) -> !torch.vtensor<[?,12],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[3,3,2,2],f32> -> tensor<3x3x2x2xf32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0], [1, 2, 3]] : tensor<3x3x2x2xf32> into tensor<3x12xf32>
// CHECK:           %[[DYNAMIC:.*]] = tensor.cast %[[COLLAPSED]] : tensor<3x12xf32> to tensor<?x12xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[DYNAMIC]] : tensor<?x12xf32> -> !torch.vtensor<[?,12],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,12],f32>

func.func @torch.aten.flatten.using_ints$flatten_back(%arg0: !torch.vtensor<[3,3,2,2],f32>) -> !torch.vtensor<[?,12],f32> {
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %0 = torch.aten.flatten.using_ints %arg0, %int1, %int-1 : !torch.vtensor<[3,3,2,2],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,12],f32>
  return %0 : !torch.vtensor<[?,12],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints$rank0(
// CHECK-SAME:                                           %[[TENSOR:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[TENSOR]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[COLLAPSED:.*]] = tensor.expand_shape %[[BUILTIN_TENSOR]] [] output_shape [1] : tensor<f32> into tensor<1xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[COLLAPSED]] : tensor<1xf32> -> !torch.vtensor<[1],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[1],f32>

func.func @torch.aten.flatten.using_ints$rank0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.flatten.using_ints %arg0, %int0, %int0 : !torch.vtensor<[],f32>, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}
