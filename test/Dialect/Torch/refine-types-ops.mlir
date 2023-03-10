// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// This file is for tests for individual ops that require a new transfer
// function (i.e. new code called from visitOperation).

// -----
// CHECK-LABEL:   func.func @aten.arange.start$int64_dtype(
// CHECK-SAME:                    %[[START:.*]]: !torch.int,
// CHECK-SAME:                    %[[END:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T:.*]] = torch.aten.arange.start
// CHECK-SAME:         %[[START]], %[[END]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:         !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none
// CHECK-SAME:         -> !torch.vtensor<*,si64>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.vtensor<*,si64> to !torch.vtensor
// CHECK:           return %[[RET]] : !torch.vtensor
func.func @aten.arange.start$int64_dtype(%start: !torch.int, %end: !torch.int) -> !torch.vtensor {
  %none = torch.constant.none
  %ret = torch.aten.arange.start %start, %end, %none, %none, %none, %none: !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @aten.arange.start$float32_dtype(
// CHECK-SAME:                    %[[START:.*]]: !torch.float,
// CHECK-SAME:                    %[[END:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T:.*]] = torch.aten.arange.start
// CHECK-SAME:         %[[START]], %[[END]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:         !torch.float, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none
// CHECK-SAME:         -> !torch.vtensor<*,f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[RET]] : !torch.vtensor
func.func @aten.arange.start$float32_dtype(%start: !torch.float, %end: !torch.int) -> !torch.vtensor {
  %none = torch.constant.none
  %ret = torch.aten.arange.start %start, %end, %none, %none, %none, %none: !torch.float, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @aten.arange.start$specified_dtype(
// CHECK-SAME:                                                    %[[END:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T:.*]] = torch.aten.arange
// CHECK-SAME:         %[[END]], %[[CST6]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:         !torch.int, !torch.int, !torch.none, !torch.none, !torch.none
// CHECK-SAME:         -> !torch.vtensor<*,f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK:           return %[[RET]] : !torch.vtensor
func.func @aten.arange.start$specified_dtype(%end: !torch.int) -> !torch.vtensor {
  %int6 = torch.constant.int 6
  %none = torch.constant.none
  %ret = torch.aten.arange %end, %int6, %none, %none, %none: !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

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
// CHECK-LABEL:   func.func @aten.sum.dim_IntList(
// CHECK-SAME:                                       %[[T:.*]]: !torch.vtensor<*,si64>) -> !torch.vtensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[DIMLIST:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT_NEG1]]
// CHECK-SAME:        : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[RET:.*]] = torch.aten.sum.dim_IntList %[[T]], %[[DIMLIST]], %[[FALSE]], %[[NONE]]
// CHECK-SAME:        : !torch.vtensor<*,si64>, !torch.list<int>, !torch.bool, !torch.none
// CHECK-SAME:        -> !torch.vtensor<*,si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<*,si64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @aten.sum.dim_IntList(%t: !torch.vtensor<*,si64>) -> !torch.vtensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int-1 = torch.constant.int -1
  %dimList = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
  %ret = torch.aten.sum.dim_IntList %t, %dimList, %false, %none : !torch.vtensor<*,si64>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @aten.any.dim(
// CHECK-SAME:                               %[[T:.*]]: !torch.vtensor<*,i1>) -> !torch.vtensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[RET:.*]] = torch.aten.any.dim %[[T]], %[[INT_NEG1]], %[[FALSE]] : !torch.vtensor<*,i1>, !torch.int, !torch.bool -> !torch.vtensor<*,i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<*,i1> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @aten.any.dim(%t: !torch.vtensor<*,i1>) -> !torch.vtensor {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.any.dim %t, %int-1, %false : !torch.vtensor<*,i1>, !torch.int, !torch.bool -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @aten.any(
// CHECK-SAME:                           %[[T:.*]]: !torch.vtensor<*,i1>) -> !torch.vtensor {
// CHECK:           %[[RET:.*]] = torch.aten.any %[[T]] : !torch.vtensor<*,i1> -> !torch.vtensor<*,i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<*,i1> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor
func.func @aten.any(%t: !torch.vtensor<*,i1>) -> !torch.vtensor {
  %ret = torch.aten.any %t: !torch.vtensor<*,i1> -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.zeros(
// CHECK-SAME:        %[[DIM0:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[ZEROS:.*]] = torch.aten.zeros %[[SIZES]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[ZEROS]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.zeros(%dim0: !torch.int) -> !torch.tensor {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %sizesList = torch.prim.ListConstruct %dim0, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %ret = torch.aten.zeros %sizesList, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
  return %ret : !torch.tensor
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
// CHECK-LABEL:   func.func @torch.aten.tensor.float(
// CHECK-SAME:                                          %[[t:.*]]: !torch.float) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor.float %[[t]], %[[NONE]], %[[NONE]], %[[FALSE]] : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.tensor.float(%t: !torch.float) -> !torch.tensor {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %ret = torch.aten.tensor.float %t, %none, %none, %false : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.tensor
  return %ret : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @torch.aten.tensor.float$specified_dtype(
// CHECK-SAME:                                          %[[t:.*]]: !torch.float) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[CST11:.*]] = torch.constant.int 11
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor.float %[[t]], %[[CST11]], %[[NONE]], %[[FALSE]] : !torch.float, !torch.int, !torch.none, !torch.bool -> !torch.tensor<*,i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<*,i1> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.tensor.float$specified_dtype(%t: !torch.float) -> !torch.tensor {
  %none = torch.constant.none
  %int11 = torch.constant.int 11
  %false = torch.constant.bool false
  %ret = torch.aten.tensor.float %t, %int11, %none, %false : !torch.float, !torch.int, !torch.none, !torch.bool -> !torch.tensor
  return %ret : !torch.tensor
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
// CHECK-LABEL:  func.func @torch.aten.Matmul.Broadcast.Matrix(
// CHECK-SAME:                                            %[[LHS:.*]]: !torch.vtensor<*,f32>,
// CHECK-SAME:                                            %[[RHS:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
// CHECK:           %[[MUL:.*]] = torch.aten.matmul %[[LHS]], %[[RHS]] : !torch.vtensor<*,f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[MUL]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.Matmul.Broadcast.Matrix(%arg0: !torch.vtensor<*,f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<*,f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:  func.func @torch.aten.Matmul.Broadcast.Vector(
// CHECK-SAME:                                            %[[LHS:.*]]: !torch.vtensor<*,f32>,
// CHECK-SAME:                                            %[[RHS:.*]]: !torch.vtensor<*,f32>) -> !torch.tensor {
// CHECK:           %[[MUL:.*]] = torch.aten.matmul %[[LHS]], %[[RHS]] : !torch.vtensor<*,f32>, !torch.vtensor<*,f32> -> !torch.tensor<*,f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[MUL]] : !torch.tensor<*,f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func.func @torch.aten.Matmul.Broadcast.Vector(%arg0: !torch.vtensor<*,f32>, %arg1: !torch.vtensor<*,f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<*,f32>, !torch.vtensor<*,f32> -> !torch.tensor
  return %0 : !torch.tensor
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
