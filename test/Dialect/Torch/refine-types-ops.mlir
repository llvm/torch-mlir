// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// This file is for tests for individual ops that require a new transfer
// function (i.e. new code called from visitOperation).

// -----
// CHECK-LABEL:   func.func @torch.aten.linear(
// CHECK-SAME:                            %[[ARG0:.*]]: !torch.vtensor<[?,3],f32>,
// CHECK-SAME:                            %[[ARG1:.*]]: !torch.vtensor<[5,3],f32>,
// CHECK-SAME:                            %[[ARG2:.*]]: !torch.vtensor<[5],f32>) -> !torch.vtensor {
// CHECK:           %[[LINEAR:.*]] = torch.aten.linear %[[ARG0]], %[[ARG1]], %[[ARG2]] : !torch.vtensor<[?,3],f32>, !torch.vtensor<[5,3],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<*,f32>
// CHECK:           %[[RESULT:.*]] = torch.tensor_static_info_cast %[[LINEAR]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @torch.aten.linear(%arg0: !torch.vtensor<[?,3],f32>, %arg1: !torch.vtensor<[5,3],f32>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor {
  %1 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[5,3],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.type_as(
// CHECK-SAME:                                     %[[INPUT:.*]]: !torch.tensor<[?],si64>,
// CHECK-SAME:                                     %[[OTHER:.*]]: !torch.tensor<[?,2],f32>) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten.type_as %[[INPUT]], %[[OTHER]] : !torch.tensor<[?],si64>, !torch.tensor<[?,2],f32> -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.type_as(%self: !torch.tensor<[?], si64>, %other: !torch.tensor<[?,2],f32>) -> !torch.tensor {
  %ret = torch.aten.type_as %self, %other : !torch.tensor<[?], si64>, !torch.tensor<[?,2],f32> -> !torch.tensor
  return %ret: !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.cat(
// CHECK-SAME:                                 %[[T1:.*]]: !torch.tensor<[?,1,4],f32>,
// CHECK-SAME:                                 %[[T2:.*]]: !torch.tensor<[2,3,4],f32>) -> !torch.tensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[TENSORS:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]] : (!torch.tensor<[?,1,4],f32>, !torch.tensor<[2,3,4],f32>) -> !torch.list<tensor>
// CHECK:           %[[RET:.*]] = torch.aten.cat %[[TENSORS]], %[[INT1]] : !torch.list<tensor>, !torch.int -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.cat(%t0: !torch.tensor<[?,1,4], f32>, %t1: !torch.tensor<[2,3,4], f32>) -> !torch.tensor {
  %int1 = torch.constant.int 1
  %tensorList = torch.prim.ListConstruct %t0, %t1: (!torch.tensor<[?,1,4], f32>, !torch.tensor<[2,3,4], f32>) -> !torch.list<tensor>
  %ret = torch.aten.cat %tensorList, %int1 : !torch.list<tensor>, !torch.int -> !torch.tensor
  return %ret : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.cat$promote_type(
// CHECK-SAME:                                 %[[T1:.*]]: !torch.tensor<[2,1,4],i1>,
// CHECK-SAME:                                 %[[T2:.*]]: !torch.tensor<[2,3,4],si64>) -> !torch.tensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[TENSORS:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]] : (!torch.tensor<[2,1,4],i1>, !torch.tensor<[2,3,4],si64>) -> !torch.list<tensor>
// CHECK:           %[[RET:.*]] = torch.aten.cat %[[TENSORS]], %[[INT1]] : !torch.list<tensor>, !torch.int -> !torch.tensor<*,si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.cat$promote_type(%t0: !torch.tensor<[2,1,4], i1>, %t1: !torch.tensor<[2,3,4], si64>) -> !torch.tensor {
  %int1 = torch.constant.int 1
  %tensorList = torch.prim.ListConstruct %t0, %t1: (!torch.tensor<[2,1,4], i1>, !torch.tensor<[2,3,4], si64>) -> !torch.list<tensor>
  %ret = torch.aten.cat %tensorList, %int1 : !torch.list<tensor>, !torch.int -> !torch.tensor
  return %ret : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten._shape_as_tensor(
// CHECK-SAME:                                 %[[INPUT:.*]]: !torch.tensor<[?,1,4],f32>) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten._shape_as_tensor %[[INPUT]] : !torch.tensor<[?,1,4],f32> -> !torch.tensor<*,si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten._shape_as_tensor(%input: !torch.tensor<[?,1,4], f32>) -> !torch.tensor {
  %ret= torch.aten._shape_as_tensor %input : !torch.tensor<[?,1,4], f32> -> !torch.tensor
  return %ret : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten._shape_as_tensor$unknown_input_shape(
// CHECK-SAME:                                 %[[INPUT:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten._shape_as_tensor %[[INPUT]] : !torch.tensor -> !torch.tensor<*,si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten._shape_as_tensor$unknown_input_shape(%input: !torch.tensor) -> !torch.tensor {
  %ret= torch.aten._shape_as_tensor %input : !torch.tensor -> !torch.tensor
  return %ret : !torch.tensor
}

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

// -----
// CHECK-LABEL: func.func @torch.aten.to.dtype(
// CHECK-SAME:                            %[[ARG:.*]]: !torch.tensor<[?,?],f32>) -> !torch.tensor
// CHECK:           %[[TODTYPE:.*]] = torch.aten.to.dtype
// CHECK-SAME:          %[[ARG]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME:          !torch.tensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none
// CHECK-SAME:          -> !torch.tensor<*,si64>
// CHECK-NEXT:      %[[RES:.*]] = torch.tensor_static_info_cast %[[TODTYPE]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK-NEXT:      return %[[RES]] : !torch.tensor
func.func @torch.aten.to.dtype(%arg0: !torch.tensor<[?,?],f32>) -> !torch.tensor{
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int4 = torch.constant.int 4
  %0 = torch.aten.to.dtype %arg0, %int4, %false, %false, %none : !torch.tensor<[?,?],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:  func.func @torch.prim.NumToTensor.Scalar(
// CHECK-SAME:                                       %[[SELF:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[NTT:.*]] = torch.prim.NumToTensor.Scalar %[[SELF]] : !torch.int -> !torch.tensor<*,si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[NTT]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.prim.NumToTensor.Scalar(%arg0: !torch.int) -> !torch.tensor {
  %0 = torch.prim.NumToTensor.Scalar %arg0: !torch.int -> !torch.tensor
  return %0: !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.tensor(
// CHECK-SAME:        %[[DATA:.*]]: !torch.list<list<float>>) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor %[[DATA]], %[[NONE]], %[[NONE]], %[[FALSE]]
// CHECK-SAME:        : !torch.list<list<float>>, !torch.none, !torch.none, !torch.bool
// CHECK-SAME:        -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.tensor(%t: !torch.list<list<float>>) -> !torch.tensor {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %ret = torch.aten.tensor %t, %none, %none, %false : !torch.list<list<float>>, !torch.none, !torch.none, !torch.bool -> !torch.tensor
  return %ret : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.tensor$specified_dtype(
// CHECK-SAME:        %[[DATA:.*]]: !torch.list<list<float>>) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor %[[DATA]], %[[INT4]], %[[NONE]], %[[FALSE]] : !torch.list<list<float>>, !torch.int, !torch.none, !torch.bool -> !torch.tensor<*,si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.tensor$specified_dtype(%t: !torch.list<list<float>>) -> !torch.tensor {
  %none = torch.constant.none
  %int4 = torch.constant.int 4
  %false = torch.constant.bool false
  %ret = torch.aten.tensor %t, %int4, %none, %false : !torch.list<list<float>>, !torch.int, !torch.none, !torch.bool -> !torch.tensor
  return %ret : !torch.tensor
}
