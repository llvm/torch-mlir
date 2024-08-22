// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @matmul_no_decompose
// CHECK:           torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_no_decompose(%arg0: !torch.vtensor<[?,?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}


// -----

// CHECK-LABEL:   func.func @matmul_decompose_2d
// CHECK:           torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
func.func @matmul_decompose_2d(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @matmul_decompose_3d(
// CHECK:           torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_decompose_3d(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:  func.func @torch.aten.type_as$basic(
// CHECK-SAME:                                %[[ARG_0:.*]]: !torch.tensor, %[[ARG_1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK-DAG:      %[[FALSE:.*]] = torch.constant.bool false
// CHECK-DAG:      %[[NONE:.*]] = torch.constant.none
// CHECK:          %[[DTYPE:.*]] = torch.prim.dtype %[[ARG_1]] : !torch.tensor -> !torch.int
// CHECK:          %[[VAR:.*]] = torch.aten.to.dtype %[[ARG_0]], %[[DTYPE]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.tensor, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
// CHECK:          return %[[VAR]] : !torch.tensor
func.func @torch.aten.type_as$basic(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor {
  %0 = torch.aten.type_as %arg0, %arg1 : !torch.tensor, !torch.tensor -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @torch.aten.type_as$fold(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.tensor<[?],f16>, %[[ARG_1:.*]]: !torch.tensor<[?,?],f16>) -> !torch.tensor<[?],f16> {
// CHECK:           return %[[ARG_0]] : !torch.tensor<[?],f16>
func.func @torch.aten.type_as$fold(%arg0: !torch.tensor<[?], f16>, %arg1: !torch.tensor<[?,?],f16>) -> !torch.tensor<[?],f16> {
  %0 = torch.aten.type_as %arg0, %arg1 : !torch.tensor<[?], f16>, !torch.tensor<[?,?],f16> -> !torch.tensor<[?], f16>
  return %0 : !torch.tensor<[?], f16>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.one_hot$fold(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[3],si64>, %arg1: !torch.int) -> !torch.vtensor<[3,?],si64> {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[ARANGE:.*]] = torch.aten.arange.start_step %[[INT0]], %arg1, %[[INT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ARG_0]], %[[INT1]] : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3,1],si64>
// CHECK:           %[[EQ:.*]] = torch.aten.eq.Tensor %[[UNSQUEEZE]], %[[ARANGE]] : !torch.vtensor<[3,1],si64>, !torch.vtensor<[?],si64> -> !torch.vtensor<[3,?],i1>
// CHECK:           %[[RESULT:.*]] = torch.aten.to.dtype %[[EQ]], %[[INT4]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[3,?],i1>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[3,?],si64>
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor<[3,?],si64>
func.func @torch.aten.one_hot$fold(%arg0: !torch.vtensor<[3],si64>, %arg1: !torch.int) -> !torch.vtensor<[3,?],si64> {
  %0 = torch.aten.one_hot %arg0, %arg1 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3,?],si64>
  return %0 : !torch.vtensor<[3,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, %[[ARG_1:.*]]: !torch.vtensor<[1],f32>,
// CHECK-SAME:                                 %[[ARG_2:.*]]: !torch.vtensor<[1],si32>, %[[ARG_3:.*]]: !torch.vtensor<[1],si64>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[CONST1:.*]] = torch.constant.int 127
// CHECK:           %[[CONST2:.*]] = torch.constant.int -128
// CHECK:           %[[RESULT:.*]] = torch.aten.fake_quantize_per_tensor_affine.tensor_qparams %[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[CONST2]], %[[CONST1]] : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[1],f32>, %arg2: !torch.vtensor<[1],si32>, %arg3: !torch.vtensor<[1],si64>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int127 = torch.constant.int 127
  %int-128 = torch.constant.int -128
  %0:2 = torch.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams %arg0, %arg1, %arg2, %arg3, %int-128, %int127 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],i1>
  return %0#0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fake_quantize_per_channel_affine_cachemask(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>, %[[ARG_1:.*]]: !torch.vtensor<[?],f32>,
// CHECK-SAME:                                 %[[ARG_2:.*]]: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[CONST0:.*]] = torch.constant.int 0
// CHECK:           %[[CONST1:.*]] = torch.constant.int 127
// CHECK:           %[[CONST2:.*]] = torch.constant.int -128
// CHECK:           %[[RESULT:.*]] = torch.aten.fake_quantize_per_channel_affine %[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[CONST0]], %[[CONST2]], %[[CONST1]] : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.fake_quantize_per_channel_affine_cachemask(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?],f32>, %arg2: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int0 = torch.constant.int 0
  %int127 = torch.constant.int 127
  %int-128 = torch.constant.int -128
  %0:2 = torch.aten.fake_quantize_per_channel_affine_cachemask %arg0, %arg1, %arg2, %int0, %int-128, %int127 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?],f32>, !torch.vtensor<[?],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],i1>
  return %0#0 : !torch.vtensor<[?,?,?,?],f32>
}
