// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// This file is for tests for individual ops that require a new transfer
// function (i.e. new code called from visitOperation).

// -----
// CHECK-LABEL:   func.func @torch.aten.embedding(
// CHECK-SAME:                                       %[[INPUT:.*]]: !torch.tensor<[104,512],f32>,
// CHECK-SAME:                                       %[[INDEXES:.*]]: !torch.tensor<[2,3],si64>) -> !torch.tensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[PADDING_IDX:.*]] = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.aten.embedding %[[INPUT]], %[[INDEXES]], %[[PADDING_IDX]], %[[FALSE]], %[[FALSE]] : !torch.tensor<[104,512],f32>, !torch.tensor<[2,3],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.embedding(%weight: !torch.tensor<[104,512],f32>, %indices: !torch.tensor<[2,3], si64>) -> !torch.tensor {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %ret = torch.aten.embedding %weight, %indices, %int1, %false, %false : !torch.tensor<[104,512],f32>, !torch.tensor<[2,3], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.tensor
  return %ret: !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.softmax.int(
// CHECK-SAME:                                 %[[T:.*]]: !torch.tensor<[2,3],f32>,
// CHECK-SAME:                                 %[[DIM:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[DTYPE:.*]] = torch.constant.none
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[T]], %[[DIM]], %[[DTYPE]] : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<*,f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[RET]] : !torch.tensor
func.func @torch.aten.softmax.int(%t: !torch.tensor<[2,3],f32>, %dim: !torch.int) -> !torch.tensor {
  %none = torch.constant.none
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor
  return %ret : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.softmax.int$specified_dtype(
// CHECK-SAME:                                                 %[[T:.*]]: !torch.tensor<[2,3],f32>,
// CHECK-SAME:                                                 %[[DIM:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[DTYPE:.*]] = torch.constant.int 4
// CHECK:           %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[T]], %[[DIM]], %[[DTYPE]] : !torch.tensor<[2,3],f32>, !torch.int, !torch.int -> !torch.tensor<*,si64>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[SOFTMAX]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK:           return %[[RET]] : !torch.tensor
func.func @torch.aten.softmax.int$specified_dtype(%t: !torch.tensor<[2,3],f32>, %dim: !torch.int) -> !torch.tensor {
  %int4 = torch.constant.int 4
  %ret = torch.aten.softmax.int %t, %dim, %int4: !torch.tensor<[2,3],f32>, !torch.int, !torch.int -> !torch.tensor
  return %ret : !torch.tensor
}

