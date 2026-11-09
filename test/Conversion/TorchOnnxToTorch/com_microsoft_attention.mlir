// RUN: torch-mlir-opt <%s -convert-torch-onnx-to-torch --split-input-file | FileCheck %s --implicit-check-not=torch.operator

// Test com.microsoft::Attention decomposition.
//   input   [1, 4, 8]
//   weights [8, 24]  bias [24]
//   num_heads = 2, qkv_hidden_sizes = [8, 8, 8], scale = 0.5
//   output  [1, 4, 8]

// CHECK-LABEL: func.func @test_com_microsoft_attention(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention without the optional bias input.

// CHECK-LABEL: func.func @test_com_microsoft_attention_no_bias(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
func.func @test_com_microsoft_attention_no_bias(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[MATMUL]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[MATMUL]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[MATMUL]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with unidirectional (causal) masking enabled. The causal mask is
// built from a strictly-upper-triangular ones tensor scaled by
// mask_filter_value and added to the scores before softmax.

// CHECK-LABEL: func.func @test_com_microsoft_attention_unidirectional(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_unidirectional(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_1:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[INT4_2:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_3:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[INT4_2]], %[[INT4_3]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FLOAT1:[^ ]+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[FULL:.*]] = torch.aten.full %[[LIST_3]], %[[FLOAT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[INT1_1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[TRIU:.*]] = torch.aten.triu %[[FULL]], %[[INT1_1]] : !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[FLOATM10000:[^ ]+]] = torch.constant.float -1.000000e+04
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Scalar %[[TRIU]], %[[FLOATM10000]] : !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Tensor %[[MUL]], %[[MUL_1]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[ADD_1]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with a 2D raw mask_index (batch, total_seq). The mask is converted
// to an additive bias (keep -> 0, mask -> mask_filter_value), reshaped to
// [batch, 1, 1, total_seq] and added to the scores.

// CHECK-LABEL: func.func @test_com_microsoft_attention_mask_index_2d(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[MASK:[a-z0-9]+]]: !torch.vtensor<[1,4],si32>
func.func @test_com_microsoft_attention_mask_index_2d(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %mask: !torch.vtensor<[1,4],si32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[FALSE:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[INT6:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[FALSE_1:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[TO:.*]] = torch.aten.to.dtype %[[MASK]], %[[INT6]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[1,4],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[FLOAT10000:[^ ]+]] = torch.constant.float 1.000000e+04
  // CHECK: %[[FLOATM10000:[^ ]+]] = torch.constant.float -1.000000e+04
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Scalar %[[TO]], %[[FLOAT10000]] : !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Scalar %[[MUL_1]], %[[FLOATM10000]], %[[INT1]] : !torch.vtensor<[1,4],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ADD_1]], %[[INT1]] : !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[1,1,4],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE]], %[[INT1]] : !torch.vtensor<[1,1,4],f32>, !torch.int -> !torch.vtensor<[1,1,1,4],f32>
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.Tensor %[[MUL]], %[[UNSQUEEZE_1]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,1,1,4],f32>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[ADD_2]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %mask) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[1,4],si32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with an attention_bias input (index 5). mask_index is passed as a
// none value so the bias slot is reached. The bias is added directly to the
// scores prior to softmax.

// CHECK-LABEL: func.func @test_com_microsoft_attention_attention_bias(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[ATTN_BIAS:[a-z0-9]+]]: !torch.vtensor<[1,2,4,4],f32>
func.func @test_com_microsoft_attention_attention_bias(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %attn_bias: !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Tensor %[[MUL]], %[[ATTN_BIAS]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[ADD_1]], %[[INT3]], %[[NONE_1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %none = torch.constant.none
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %none, %attn_bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.none, !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention on f16 tensors. The decomposition preserves the input element type
// throughout, so all intermediate tensors are f16.

// CHECK-LABEL: func.func @test_com_microsoft_attention_f16(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f16>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f16>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f16>
func.func @test_com_microsoft_attention_f16(%input: !torch.vtensor<[1,4,8],f16>, %weights: !torch.vtensor<[8,24],f16>, %bias: !torch.vtensor<[24],f16>) -> !torch.vtensor<[1,4,8],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f16>, !torch.vtensor<[8,24],f16> -> !torch.vtensor<[1,4,24],f16>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f16>, !torch.vtensor<[24],f16>, !torch.int -> !torch.vtensor<[1,4,24],f16>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f16>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f16>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f16>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f16>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f16>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f16>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f16>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f16>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f16>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f16>, !torch.vtensor<[1,2,4,4],f16> -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f16>, !torch.float -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f16>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f16>, !torch.vtensor<[1,2,4,4],f16> -> !torch.vtensor<[1,2,4,4],f16>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f16>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f16>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f16>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f16>, !torch.vtensor<[8,24],f16>, !torch.vtensor<[24],f16>) -> !torch.vtensor<[1,4,8],f16>
  return %0 : !torch.vtensor<[1,4,8],f16>
}

// -----

// Attention with an explicit scale = 0.0. The com.microsoft spec allows a
// user-provided scale of 0.0, so 0.0 must NOT be treated as "unset". This test
// pins that behavior: with num_heads = 1 the head_size is 8, so an "unset"
// scale would produce 1/sqrt(8) = 0.353..., whereas an explicit scale = 0.0
// must be honored and emit a mul by exactly 0.0.

// CHECK-LABEL: func.func @test_com_microsoft_attention_scale_zero(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_scale_zero(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT1_1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT8_1:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8_2:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8_2]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8_2]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,1,4,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,1,4,8],f32>, !torch.vtensor<[1,1,8,4],f32> -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[FLOAT0:[^ ]+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0]] : !torch.vtensor<[1,1,4,4],f32>, !torch.float -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,1,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,8],f32> -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,1,4,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 1 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 0.000000e+00 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with scale omitted. The scale must default to 1/sqrt(head_size).
// With num_heads = 1 and qkv_hidden_sizes = [8, 8, 8] the head_size is 8, so
// the emitted multiplier is 1/sqrt(8) = 0.35355339...

// CHECK-LABEL: func.func @test_com_microsoft_attention_scale_default(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_scale_default(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT1_1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT8_1:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8_2:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8_2]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8_2]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT1_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,1,4,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,1,4,8],f32>, !torch.vtensor<[1,1,8,4],f32> -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[FLOAT0_353553:[^ ]+]] = torch.constant.float 0.35355339059327373
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_353553]] : !torch.vtensor<[1,1,4,4],f32>, !torch.float -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,1,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[1,1,4,8],f32> -> !torch.vtensor<[1,1,4,8],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,1,4,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,8],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,1,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 1 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64]} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with a past KV-cache input and a present output.
// past shape: [2, 1, 2, 3, 4]  (2=K/V, batch=1, heads=2, past_seq=3, head=4)
// input shape: [1, 2, 16]  (batch=1, seq=2, input_hidden=16 -> qkv=3*8=24)
// output shape: [1, 2, 8], present shape: [2, 1, 2, 5, 4]
//   (total_seq = past_seq(3) + seq(2) = 5)
//
// The lowering must:
//   - select(past, 0, 0) -> pastK [1, 2, 3, 4]
//   - select(past, 0, 1) -> pastV [1, 2, 3, 4]
//   - cat([pastK, kHeads], dim=2) -> kHeads [1, 2, 5, 4]
//   - cat([pastV, vHeads], dim=2) -> vHeads [1, 2, 5, 4]
//   - scores shape [1, 2, 2, 5]
//   - unsqueeze kHeads/vHeads on dim 0 -> [1, 1, 2, 5, 4]
//   - cat -> present [2, 1, 2, 5, 4]

// CHECK-LABEL: func.func @test_com_microsoft_attention_past_present(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,2,16],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[16,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[PAST:[a-z0-9]+]]: !torch.vtensor<[2,1,2,3,4],f32>
func.func @test_com_microsoft_attention_past_present(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32> -> !torch.vtensor<[1,2,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,2,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT0]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[SELECT_1:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SELECT]], %[[TRANSPOSE_1]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT:.*]] = torch.aten.cat %[[LIST_3]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[LIST_4:[^ ]+]] = torch.prim.ListConstruct %[[SELECT_1]], %[[TRANSPOSE_2]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_1:.*]] = torch.aten.cat %[[LIST_4]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[CAT]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,5],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,4,5],f32> -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,2,5],f32>, !torch.float -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE_1]] : !torch.vtensor<[1,2,2,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[CAT_1]] : !torch.vtensor<[1,2,2,5],f32>, !torch.vtensor<[1,2,5,4],f32> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[CAT]], %[[INT0]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[CAT_1]], %[[INT0]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: %[[LIST_5:[^ ]+]] = torch.prim.ListConstruct %[[UNSQUEEZE]], %[[UNSQUEEZE_1]] : (!torch.vtensor<[1,1,2,5,4],f32>, !torch.vtensor<[1,1,2,5,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_2:.*]] = torch.aten.cat %[[LIST_5]], %[[INT0]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,1,2,5,4],f32>
  // CHECK: return %[[FLATTEN]], %[[CAT_2]] : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// Attention with do_rotary = 1. With head_size = 4 and rotary_embedding_dim = 0
// (default, meaning rotate all 4 elements), the lowering must emit:
//   - arange to build freq indices
//   - exp/cos/sin for the RoPE tables
//   - slice x1/x2 and the rotate-combine (mul, sub, add, cat)
// applied to both Q and K heads before the attention scores.

// CHECK-LABEL: func.func @test_com_microsoft_attention_do_rotary(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_do_rotary(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_1:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[FALSE:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[INT4_2:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_3:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT2_2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4_4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[FLOAT4:[^ ]+]] = torch.constant.float 4.000000e+00
  // CHECK: %[[FLOAT0:[^ ]+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[FLOAT4_1:[^ ]+]] = torch.constant.float 4.000000e+00
  // CHECK: %[[FLOAT2:[^ ]+]] = torch.constant.float 2.000000e+00
  // CHECK: %[[ARANGE:.*]] = torch.aten.arange.start_step %[[FLOAT0]], %[[FLOAT4_1]], %[[FLOAT2]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.float, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],f32>
  // CHECK: %[[DIV:.*]] = torch.aten.div.Scalar %[[ARANGE]], %[[FLOAT4]] : !torch.vtensor<[2],f32>, !torch.float -> !torch.vtensor<[2],f32>
  // CHECK: %[[FLOAT9_21034:[^ ]+]] = torch.constant.float 9.2103403719761836
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[DIV]], %[[FLOAT9_21034]] : !torch.vtensor<[2],f32>, !torch.float -> !torch.vtensor<[2],f32>
  // CHECK: %[[NEG:.*]] = torch.aten.neg %[[MUL]] : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  // CHECK: %[[EXP:.*]] = torch.aten.exp %[[NEG]] : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  // CHECK: %[[FLOAT0_1:[^ ]+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[FLOAT1:[^ ]+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[INT6:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[ARANGE_1:.*]] = torch.aten.arange.start_step %[[FLOAT0_1]], %[[INT4_3]], %[[FLOAT1]], %[[INT6]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ARANGE_1]], %[[INT1]] : !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[EXP]], %[[INT0]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Tensor %[[UNSQUEEZE]], %[[UNSQUEEZE_1]] : !torch.vtensor<[4,1],f32>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[4,2],f32>
  // CHECK: %[[COS:.*]] = torch.aten.cos %[[MUL_1]] : !torch.vtensor<[4,2],f32> -> !torch.vtensor<[4,2],f32>
  // CHECK: %[[SIN:.*]] = torch.aten.sin %[[MUL_1]] : !torch.vtensor<[4,2],f32> -> !torch.vtensor<[4,2],f32>
  // CHECK: %[[INT6_1:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[TO:.*]] = torch.aten.to.dtype %[[COS]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[4,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[4,2],f32>
  // CHECK: %[[TO_1:.*]] = torch.aten.to.dtype %[[SIN]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[4,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[4,2],f32>
  // CHECK: %[[UNSQUEEZE_2:.*]] = torch.aten.unsqueeze %[[TO]], %[[INT0]] : !torch.vtensor<[4,2],f32>, !torch.int -> !torch.vtensor<[1,4,2],f32>
  // CHECK: %[[UNSQUEEZE_3:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE_2]], %[[INT0]] : !torch.vtensor<[1,4,2],f32>, !torch.int -> !torch.vtensor<[1,1,4,2],f32>
  // CHECK: %[[UNSQUEEZE_4:.*]] = torch.aten.unsqueeze %[[TO_1]], %[[INT0]] : !torch.vtensor<[4,2],f32>, !torch.int -> !torch.vtensor<[1,4,2],f32>
  // CHECK: %[[UNSQUEEZE_5:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE_4]], %[[INT0]] : !torch.vtensor<[1,4,2],f32>, !torch.int -> !torch.vtensor<[1,1,4,2],f32>
  // CHECK: %[[SLICE_3:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT0]], %[[INT2_2]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[SLICE_4:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT2_2]], %[[INT4_4]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_2:.*]] = torch.aten.mul.Tensor %[[SLICE_4]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_3:.*]] = torch.aten.mul.Tensor %[[SLICE_3]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Tensor %[[MUL_3]], %[[MUL_2]], %[[INT1]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_4:.*]] = torch.aten.mul.Tensor %[[SLICE_3]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_5:.*]] = torch.aten.mul.Tensor %[[SLICE_4]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Tensor %[[MUL_5]], %[[MUL_4]], %[[INT1]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SUB]], %[[ADD_1]] : (!torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT:.*]] = torch.aten.cat %[[LIST_3]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SLICE_5:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT0]], %[[INT2_2]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[SLICE_6:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT2_2]], %[[INT4_4]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_6:.*]] = torch.aten.mul.Tensor %[[SLICE_6]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_7:.*]] = torch.aten.mul.Tensor %[[SLICE_5]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[SUB_1:.*]] = torch.aten.sub.Tensor %[[MUL_7]], %[[MUL_6]], %[[INT1]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_8:.*]] = torch.aten.mul.Tensor %[[SLICE_5]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[MUL_9:.*]] = torch.aten.mul.Tensor %[[SLICE_6]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,1,4,2],f32> -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.Tensor %[[MUL_9]], %[[MUL_8]], %[[INT1]] : !torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[LIST_4:[^ ]+]] = torch.prim.ListConstruct %[[SUB_1]], %[[ADD_2]] : (!torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_1:.*]] = torch.aten.cat %[[LIST_4]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[CAT_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[CAT]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL_10:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL_10]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with do_rotary=1 and a past KV cache. This exercises the combined
// code path: RoPE is applied with position offset pastSeq, and the present
// output must store the POST-RoPE K/V (matching ORT CUDA, which writes present
// from the rotated QKV workspace). Because present stores rotated K, there is
// only one K concat (post-RoPE cat([pastK, kRotated])) and it is reused both
// for attention and for the present output - there is no separate pre-RoPE
// cat. This is the regression guard for the "cached keys never rotated" bug:
// if present were rebuilt from an unrotated kHeadsPreRope, this single-cat
// structure would not hold.
// Model dims: batch=1, seq=2, num_heads=2, head_size=4, past_seq=3.
//   - input [1,2,8], weights [8,24], bias [24], past [2,1,2,3,4]
//   - qkv_hidden_sizes = [8,8,8], rotary_embedding_dim = 0 -> rotate all 4 dims
//   - freqRange arange.start_step with float scalars -> exp -> invFreq
//   - posRange arange.start_step(3.0, 5.0, 1.0) for positions [3,4]
//   - cos/sin computed in f32, cast to elemDtype, applied to Q/K new tokens only
//   - cat([pastK, kRotated], dim=2) -> kHeads [1,2,5,4] for attention and present
//   - output [1,2,8], present [2,1,2,5,4]

// CHECK-LABEL: func.func @test_com_microsoft_attention_rotary_with_past(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,2,16],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[16,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[PAST:[a-z0-9]+]]: !torch.vtensor<[2,1,2,3,4],f32>
func.func @test_com_microsoft_attention_rotary_with_past(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_1:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[FALSE:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32> -> !torch.vtensor<[1,2,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,2,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[INT2_2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT5:[^ ]+]] = torch.constant.int 5
  // CHECK: %[[INT2_3:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4_2:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[FLOAT4:[^ ]+]] = torch.constant.float 4.000000e+00
  // CHECK: %[[FLOAT0:[^ ]+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[FLOAT4_1:[^ ]+]] = torch.constant.float 4.000000e+00
  // CHECK: %[[FLOAT2:[^ ]+]] = torch.constant.float 2.000000e+00
  // CHECK: %[[ARANGE:.*]] = torch.aten.arange.start_step %[[FLOAT0]], %[[FLOAT4_1]], %[[FLOAT2]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.float, !torch.float, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],f32>
  // CHECK: %[[DIV:.*]] = torch.aten.div.Scalar %[[ARANGE]], %[[FLOAT4]] : !torch.vtensor<[2],f32>, !torch.float -> !torch.vtensor<[2],f32>
  // CHECK: %[[FLOAT9_21034:[^ ]+]] = torch.constant.float 9.2103403719761836
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[DIV]], %[[FLOAT9_21034]] : !torch.vtensor<[2],f32>, !torch.float -> !torch.vtensor<[2],f32>
  // CHECK: %[[NEG:.*]] = torch.aten.neg %[[MUL]] : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  // CHECK: %[[EXP:.*]] = torch.aten.exp %[[NEG]] : !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
  // CHECK: %[[FLOAT3:[^ ]+]] = torch.constant.float 3.000000e+00
  // CHECK: %[[FLOAT1:[^ ]+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[INT6:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[ARANGE_1:.*]] = torch.aten.arange.start_step %[[FLOAT3]], %[[INT5]], %[[FLOAT1]], %[[INT6]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.float, !torch.int, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ARANGE_1]], %[[INT1]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2,1],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[EXP]], %[[INT0]] : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Tensor %[[UNSQUEEZE]], %[[UNSQUEEZE_1]] : !torch.vtensor<[2,1],f32>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[2,2],f32>
  // CHECK: %[[COS:.*]] = torch.aten.cos %[[MUL_1]] : !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
  // CHECK: %[[SIN:.*]] = torch.aten.sin %[[MUL_1]] : !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
  // CHECK: %[[INT6_1:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[TO:.*]] = torch.aten.to.dtype %[[COS]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,2],f32>
  // CHECK: %[[TO_1:.*]] = torch.aten.to.dtype %[[SIN]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE_1]] : !torch.vtensor<[2,2],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,2],f32>
  // CHECK: %[[UNSQUEEZE_2:.*]] = torch.aten.unsqueeze %[[TO]], %[[INT0]] : !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[1,2,2],f32>
  // CHECK: %[[UNSQUEEZE_3:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE_2]], %[[INT0]] : !torch.vtensor<[1,2,2],f32>, !torch.int -> !torch.vtensor<[1,1,2,2],f32>
  // CHECK: %[[UNSQUEEZE_4:.*]] = torch.aten.unsqueeze %[[TO_1]], %[[INT0]] : !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[1,2,2],f32>
  // CHECK: %[[UNSQUEEZE_5:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE_4]], %[[INT0]] : !torch.vtensor<[1,2,2],f32>, !torch.int -> !torch.vtensor<[1,1,2,2],f32>
  // CHECK: %[[SLICE_3:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT0]], %[[INT2_3]], %[[INT1]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[SLICE_4:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT2_3]], %[[INT4_2]], %[[INT1]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_2:.*]] = torch.aten.mul.Tensor %[[SLICE_4]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_3:.*]] = torch.aten.mul.Tensor %[[SLICE_3]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Tensor %[[MUL_3]], %[[MUL_2]], %[[INT1]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,2,2,2],f32>, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_4:.*]] = torch.aten.mul.Tensor %[[SLICE_3]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_5:.*]] = torch.aten.mul.Tensor %[[SLICE_4]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Tensor %[[MUL_5]], %[[MUL_4]], %[[INT1]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,2,2,2],f32>, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SUB]], %[[ADD_1]] : (!torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT:.*]] = torch.aten.cat %[[LIST_3]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[SLICE_5:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT0]], %[[INT2_3]], %[[INT1]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[SLICE_6:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT2_3]], %[[INT4_2]], %[[INT1]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_6:.*]] = torch.aten.mul.Tensor %[[SLICE_6]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_7:.*]] = torch.aten.mul.Tensor %[[SLICE_5]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[SUB_1:.*]] = torch.aten.sub.Tensor %[[MUL_7]], %[[MUL_6]], %[[INT1]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,2,2,2],f32>, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_8:.*]] = torch.aten.mul.Tensor %[[SLICE_5]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[MUL_9:.*]] = torch.aten.mul.Tensor %[[SLICE_6]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,1,2,2],f32> -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.Tensor %[[MUL_9]], %[[MUL_8]], %[[INT1]] : !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,2,2,2],f32>, !torch.int -> !torch.vtensor<[1,2,2,2],f32>
  // CHECK: %[[LIST_4:[^ ]+]] = torch.prim.ListConstruct %[[SUB_1]], %[[ADD_2]] : (!torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,2,2,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_1:.*]] = torch.aten.cat %[[LIST_4]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT0]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[SELECT_1:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[LIST_5:[^ ]+]] = torch.prim.ListConstruct %[[SELECT]], %[[CAT_1]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_2:.*]] = torch.aten.cat %[[LIST_5]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[LIST_6:[^ ]+]] = torch.prim.ListConstruct %[[SELECT_1]], %[[TRANSPOSE_2]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_3:.*]] = torch.aten.cat %[[LIST_6]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[CAT_2]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,5],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[CAT]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,4,5],f32> -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL_10:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,2,5],f32>, !torch.float -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL_10]], %[[INT3]], %[[NONE_1]] : !torch.vtensor<[1,2,2,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[CAT_3]] : !torch.vtensor<[1,2,2,5],f32>, !torch.vtensor<[1,2,5,4],f32> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // present is built from the post-RoPE K concat (CAT_2 consumes the rotated
  // K heads CAT_1), not from an unrotated copy.
  // CHECK: %[[UNSQUEEZE_6:.*]] = torch.aten.unsqueeze %[[CAT_2]], %[[INT0]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: %[[UNSQUEEZE_7:.*]] = torch.aten.unsqueeze %[[CAT_3]], %[[INT0]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: %[[LIST_7:[^ ]+]] = torch.prim.ListConstruct %[[UNSQUEEZE_6]], %[[UNSQUEEZE_7]] : (!torch.vtensor<[1,1,2,5,4],f32>, !torch.vtensor<[1,1,2,5,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_4:.*]] = torch.aten.cat %[[LIST_7]], %[[INT0]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,1,2,5,4],f32>
  // CHECK: return %[[FLATTEN]], %[[CAT_4]] : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.do_rotary = 1 : si64} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// Attention with unidirectional=1 and a past KV cache. This exercises the
// causal mask with a non-zero pastSeq so that the mask shape is
// [seq, totalSeqLen] = [2, 5] and the triu diagonal = pastSeq+1 = 4.
//   - batch=1, seq=2, num_heads=2, head_size=4, past_seq=3, total_seq=5
//   - full([2,5]) -> triu(diagonal=4) -> scale by filter_value
//   - scores [1,2,2,5] + causal mask [2,5] broadcast

// CHECK-LABEL: func.func @test_com_microsoft_attention_unidirectional_with_past(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,2,16],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[16,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[PAST:[a-z0-9]+]]: !torch.vtensor<[2,1,2,3,4],f32>
func.func @test_com_microsoft_attention_unidirectional_with_past(%input: !torch.vtensor<[1,2,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_1:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32> -> !torch.vtensor<[1,2,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,2,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,2,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,2,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[INT2_2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT5:[^ ]+]] = torch.constant.int 5
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT0]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[SELECT_1:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SELECT]], %[[TRANSPOSE_1]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT:.*]] = torch.aten.cat %[[LIST_3]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[LIST_4:[^ ]+]] = torch.prim.ListConstruct %[[SELECT_1]], %[[TRANSPOSE_2]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,2,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_1:.*]] = torch.aten.cat %[[LIST_4]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,5,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[CAT]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,5],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,2,4],f32>, !torch.vtensor<[1,2,4,5],f32> -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,2,5],f32>, !torch.float -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[LIST_5:[^ ]+]] = torch.prim.ListConstruct %[[INT2_2]], %[[INT5]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FLOAT1:[^ ]+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[FULL:.*]] = torch.aten.full %[[LIST_5]], %[[FLOAT1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]], %[[NONE_1]] : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,5],f32>
  // CHECK: %[[INT4_2:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[TRIU:.*]] = torch.aten.triu %[[FULL]], %[[INT4_2]] : !torch.vtensor<[2,5],f32>, !torch.int -> !torch.vtensor<[2,5],f32>
  // CHECK: %[[FLOATM10000:[^ ]+]] = torch.constant.float -1.000000e+04
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Scalar %[[TRIU]], %[[FLOATM10000]] : !torch.vtensor<[2,5],f32>, !torch.float -> !torch.vtensor<[2,5],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Tensor %[[MUL]], %[[MUL_1]], %[[INT1]] : !torch.vtensor<[1,2,2,5],f32>, !torch.vtensor<[2,5],f32>, !torch.int -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[ADD_1]], %[[INT3]], %[[NONE_1]] : !torch.vtensor<[1,2,2,5],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,2,5],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[CAT_1]] : !torch.vtensor<[1,2,2,5],f32>, !torch.vtensor<[1,2,5,4],f32> -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,8],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[CAT]], %[[INT0]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[CAT_1]], %[[INT0]] : !torch.vtensor<[1,2,5,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,5,4],f32>
  // CHECK: %[[LIST_6:[^ ]+]] = torch.prim.ListConstruct %[[UNSQUEEZE]], %[[UNSQUEEZE_1]] : (!torch.vtensor<[1,1,2,5,4],f32>, !torch.vtensor<[1,1,2,5,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_2:.*]] = torch.aten.cat %[[LIST_6]], %[[INT0]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,1,2,5,4],f32>
  // CHECK: return %[[FLATTEN]], %[[CAT_2]] : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,2,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[2,1,2,5,4],f32>
}

// -----

// Attention where the V projection has a larger head size than Q/K. With
// qkv_hidden_sizes = [8, 8, 16] and num_heads = 2, the Q/K head_size is 4 but
// the V head_size is 8. The scores path stays at head_size 4 while the context
// path carries the v_head_size 8 through to a [1, 4, 16] output. This pins
// that the two head sizes are tracked independently.

// CHECK-LABEL: func.func @test_com_microsoft_attention_vheadsize(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,32],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[32],f32>
func.func @test_com_microsoft_attention_vheadsize(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,32],f32>, %bias: !torch.vtensor<[32],f32>) -> !torch.vtensor<[1,4,16],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,32],f32> -> !torch.vtensor<[1,4,32],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,32],f32>, !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,4,32],f32>
  // CHECK: %[[INT8_1:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT32:[^ ]+]] = torch.constant.int 32
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8_1]], %[[INT1]] : !torch.vtensor<[1,4,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8_1]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT32]], %[[INT1]] : !torch.vtensor<[1,4,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,16],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT8]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,16],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,8],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,8],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,8],f32> -> !torch.vtensor<[1,2,4,8],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,8],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,16],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,16],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 16 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,32],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[1,4,16],f32>
  return %0 : !torch.vtensor<[1,4,16],f32>
}

// -----

// Attention with qkv_hidden_sizes omitted. The weights hidden dimension (24)
// must be split evenly into three projections of 8, giving head_size 4 for
// num_heads = 2. This pins the default equal-split path.

// CHECK-LABEL: func.func @test_com_microsoft_attention_default_split(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_default_split(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Attention with a 3D mask_index [batch, seq, total_seq]. The mask is converted
// to an additive bias (keep -> 0, mask -> mask_filter_value) and unsqueezed on
// the head dim to [batch, 1, seq, total_seq] before being added to the scores.

// CHECK-LABEL: func.func @test_com_microsoft_attention_mask_index_3d(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[MASK:[a-z0-9]+]]: !torch.vtensor<[1,4,4],si32>
func.func @test_com_microsoft_attention_mask_index_3d(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>, %mask: !torch.vtensor<[1,4,4],si32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[FALSE:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[INT6:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[FALSE_1:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[TO:.*]] = torch.aten.to.dtype %[[MASK]], %[[INT6]], %[[FALSE_1]], %[[FALSE_1]], %[[NONE_1]] : !torch.vtensor<[1,4,4],si32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4,4],f32>
  // CHECK: %[[FLOAT10000:[^ ]+]] = torch.constant.float 1.000000e+04
  // CHECK: %[[FLOATM10000:[^ ]+]] = torch.constant.float -1.000000e+04
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Scalar %[[TO]], %[[FLOAT10000]] : !torch.vtensor<[1,4,4],f32>, !torch.float -> !torch.vtensor<[1,4,4],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Scalar %[[MUL_1]], %[[FLOATM10000]], %[[INT1]] : !torch.vtensor<[1,4,4],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4,4],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ADD_1]], %[[INT1]] : !torch.vtensor<[1,4,4],f32>, !torch.int -> !torch.vtensor<[1,1,4,4],f32>
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.Tensor %[[MUL]], %[[UNSQUEEZE]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,1,4,4],f32>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[ADD_2]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias, %mask) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>, !torch.vtensor<[1,4,4],si32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Dynamic batch and sequence dimensions are supported: only the hidden
// dimension (dim 2) must be static. The batch (dim 0) and sequence (dim 1)
// sizes flow through as dynamic (`?`) and the lowering still legalizes.

// CHECK-LABEL: func.func @test_com_microsoft_attention_dynamic_batch_seq(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[?,?,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_dynamic_batch_seq(%input: !torch.vtensor<[?,?,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[?,?,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[?,?,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[?,?,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[?,?,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[?,?,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[?,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[?,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[?,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,?,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[?,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[?,?,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[?,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[?,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[?,?,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[?,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[?,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[?,?,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[?,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[?,2,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,2,4,?],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[?,2,?,4],f32>, !torch.vtensor<[?,2,4,?],f32> -> !torch.vtensor<[?,2,?,?],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[?,2,?,?],f32>, !torch.float -> !torch.vtensor<[?,2,?,?],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE]] : !torch.vtensor<[?,2,?,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[?,2,?,?],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[?,2,?,?],f32>, !torch.vtensor<[?,2,?,4],f32> -> !torch.vtensor<[?,2,?,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[?,2,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[?,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[?,?,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32} : (!torch.vtensor<[?,?,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}

// -----

// Attention with do_rotary = 1 and a partial rotary_embedding_dim = 2 that is
// smaller than head_size = 4. Only the first 2 of each head's 4 elements are
// rotated; the unrotated tail (elements [2, 4)) must be sliced off and
// concatenated back after the rotation. This pins the `rotDim < head_size`
// tail-concat branch of applyRope, which the full-rotation rotary tests
// (rotary_embedding_dim = 0) never exercise.

// CHECK-LABEL: func.func @test_com_microsoft_attention_partial_rotary(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,4,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_partial_rotary(%input: !torch.vtensor<[1,4,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_1:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[FALSE:[^ ]+]] = torch.constant.bool false
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,4,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,4,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,4,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[INT4_2:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT4_3:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[INT1_1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT2_2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[FLOAT2:[^ ]+]] = torch.constant.float 2.000000e+00
  // CHECK: %[[FLOAT0:[^ ]+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[FLOAT2_1:[^ ]+]] = torch.constant.float 2.000000e+00
  // CHECK: %[[FLOAT2_2:[^ ]+]] = torch.constant.float 2.000000e+00
  // CHECK: %[[ARANGE:.*]] = torch.aten.arange.start_step %[[FLOAT0]], %[[FLOAT2_1]], %[[FLOAT2_2]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.float, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],f32>
  // CHECK: %[[DIV:.*]] = torch.aten.div.Scalar %[[ARANGE]], %[[FLOAT2]] : !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor<[1],f32>
  // CHECK: %[[FLOAT9_21034:[^ ]+]] = torch.constant.float 9.2103403719761836
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[DIV]], %[[FLOAT9_21034]] : !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor<[1],f32>
  // CHECK: %[[NEG:.*]] = torch.aten.neg %[[MUL]] : !torch.vtensor<[1],f32> -> !torch.vtensor<[1],f32>
  // CHECK: %[[EXP:.*]] = torch.aten.exp %[[NEG]] : !torch.vtensor<[1],f32> -> !torch.vtensor<[1],f32>
  // CHECK: %[[FLOAT0_1:[^ ]+]] = torch.constant.float 0.000000e+00
  // CHECK: %[[FLOAT1:[^ ]+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[INT6:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[ARANGE_1:.*]] = torch.aten.arange.start_step %[[FLOAT0_1]], %[[INT4_3]], %[[FLOAT1]], %[[INT6]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.float, !torch.int, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ARANGE_1]], %[[INT1]] : !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[EXP]], %[[INT0]] : !torch.vtensor<[1],f32>, !torch.int -> !torch.vtensor<[1,1],f32>
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Tensor %[[UNSQUEEZE]], %[[UNSQUEEZE_1]] : !torch.vtensor<[4,1],f32>, !torch.vtensor<[1,1],f32> -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[COS:.*]] = torch.aten.cos %[[MUL_1]] : !torch.vtensor<[4,1],f32> -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[SIN:.*]] = torch.aten.sin %[[MUL_1]] : !torch.vtensor<[4,1],f32> -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[INT6_1:[^ ]+]] = torch.constant.int 6
  // CHECK: %[[TO:.*]] = torch.aten.to.dtype %[[COS]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[4,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[TO_1:.*]] = torch.aten.to.dtype %[[SIN]], %[[INT6_1]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.vtensor<[4,1],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[4,1],f32>
  // CHECK: %[[UNSQUEEZE_2:.*]] = torch.aten.unsqueeze %[[TO]], %[[INT0]] : !torch.vtensor<[4,1],f32>, !torch.int -> !torch.vtensor<[1,4,1],f32>
  // CHECK: %[[UNSQUEEZE_3:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE_2]], %[[INT0]] : !torch.vtensor<[1,4,1],f32>, !torch.int -> !torch.vtensor<[1,1,4,1],f32>
  // CHECK: %[[UNSQUEEZE_4:.*]] = torch.aten.unsqueeze %[[TO_1]], %[[INT0]] : !torch.vtensor<[4,1],f32>, !torch.int -> !torch.vtensor<[1,4,1],f32>
  // CHECK: %[[UNSQUEEZE_5:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE_4]], %[[INT0]] : !torch.vtensor<[1,4,1],f32>, !torch.int -> !torch.vtensor<[1,1,4,1],f32>
  // CHECK: %[[SLICE_3:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT0]], %[[INT1_1]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[SLICE_4:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT1_1]], %[[INT2_2]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_2:.*]] = torch.aten.mul.Tensor %[[SLICE_4]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_3:.*]] = torch.aten.mul.Tensor %[[SLICE_3]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[SUB:.*]] = torch.aten.sub.Tensor %[[MUL_3]], %[[MUL_2]], %[[INT1]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,2,4,1],f32>, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_4:.*]] = torch.aten.mul.Tensor %[[SLICE_3]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_5:.*]] = torch.aten.mul.Tensor %[[SLICE_4]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.Tensor %[[MUL_5]], %[[MUL_4]], %[[INT1]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,2,4,1],f32>, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SUB]], %[[ADD_1]] : (!torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,2,4,1],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT:.*]] = torch.aten.cat %[[LIST_3]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[INT4_4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[SLICE_5:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE]], %[[INT3]], %[[INT2_2]], %[[INT4_4]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[LIST_4:[^ ]+]] = torch.prim.ListConstruct %[[CAT]], %[[SLICE_5]] : (!torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_1:.*]] = torch.aten.cat %[[LIST_4]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SLICE_6:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT0]], %[[INT1_1]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[SLICE_7:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT1_1]], %[[INT2_2]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_6:.*]] = torch.aten.mul.Tensor %[[SLICE_7]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_7:.*]] = torch.aten.mul.Tensor %[[SLICE_6]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[SUB_1:.*]] = torch.aten.sub.Tensor %[[MUL_7]], %[[MUL_6]], %[[INT1]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,2,4,1],f32>, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_8:.*]] = torch.aten.mul.Tensor %[[SLICE_6]], %[[UNSQUEEZE_5]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[MUL_9:.*]] = torch.aten.mul.Tensor %[[SLICE_7]], %[[UNSQUEEZE_3]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,1,4,1],f32> -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.Tensor %[[MUL_9]], %[[MUL_8]], %[[INT1]] : !torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,2,4,1],f32>, !torch.int -> !torch.vtensor<[1,2,4,1],f32>
  // CHECK: %[[LIST_5:[^ ]+]] = torch.prim.ListConstruct %[[SUB_1]], %[[ADD_2]] : (!torch.vtensor<[1,2,4,1],f32>, !torch.vtensor<[1,2,4,1],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_2:.*]] = torch.aten.cat %[[LIST_5]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[INT4_5:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[SLICE_8:.*]] = torch.aten.slice.Tensor %[[TRANSPOSE_1]], %[[INT3]], %[[INT2_2]], %[[INT4_5]], %[[INT1]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,2],f32>
  // CHECK: %[[LIST_6:[^ ]+]] = torch.prim.ListConstruct %[[CAT_2]], %[[SLICE_8]] : (!torch.vtensor<[1,2,4,2],f32>, !torch.vtensor<[1,2,4,2],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_3:.*]] = torch.aten.cat %[[LIST_6]], %[[INT3]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[CAT_3]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[CAT_1]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL_10:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,4,4],f32>, !torch.float -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL_10]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[1,2,4,4],f32> -> !torch.vtensor<[1,2,4,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,4,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,4,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,4,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.qkv_hidden_sizes = [8 : si64, 8 : si64, 8 : si64], torch.onnx.scale = 5.000000e-01 : f32, torch.onnx.do_rotary = 1 : si64, torch.onnx.rotary_embedding_dim = 2 : si64} : (!torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,4,8],f32>
  return %0 : !torch.vtensor<[1,4,8],f32>
}

// -----

// Dynamic sequence length together with a past KV cache. The total sequence
// length (past_seq + seq) is dynamic, so the K/V concat, the scores and the
// present output all carry a dynamic sequence dimension.

// CHECK-LABEL: func.func @test_com_microsoft_attention_dynamic_seq_with_past(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,?,16],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[16,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
// CHECK-SAME: %[[PAST:[a-z0-9]+]]: !torch.vtensor<[2,1,2,3,4],f32>
func.func @test_com_microsoft_attention_dynamic_seq_with_past(%input: !torch.vtensor<[1,?,16],f32>, %weights: !torch.vtensor<[16,24],f32>, %bias: !torch.vtensor<[24],f32>, %past: !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,?,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE_1:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,?,16],f32>, !torch.vtensor<[16,24],f32> -> !torch.vtensor<[1,?,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,?,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[SELECT:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT0]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[SELECT_1:.*]] = torch.aten.select.int %[[PAST]], %[[INT0]], %[[INT1]] : !torch.vtensor<[2,1,2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,3,4],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SELECT]], %[[TRANSPOSE_1]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,?,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT:.*]] = torch.aten.cat %[[LIST_3]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[LIST_4:[^ ]+]] = torch.prim.ListConstruct %[[SELECT_1]], %[[TRANSPOSE_2]] : (!torch.vtensor<[1,2,3,4],f32>, !torch.vtensor<[1,2,?,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_1:.*]] = torch.aten.cat %[[LIST_4]], %[[INT2]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[CAT]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,?],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,?,4],f32>, !torch.vtensor<[1,2,4,?],f32> -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,?,?],f32>, !torch.float -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[MUL]], %[[INT3]], %[[NONE_1]] : !torch.vtensor<[1,2,?,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[CAT_1]] : !torch.vtensor<[1,2,?,?],f32>, !torch.vtensor<[1,2,?,4],f32> -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[CAT]], %[[INT0]] : !torch.vtensor<[1,2,?,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,?,4],f32>
  // CHECK: %[[UNSQUEEZE_1:.*]] = torch.aten.unsqueeze %[[CAT_1]], %[[INT0]] : !torch.vtensor<[1,2,?,4],f32>, !torch.int -> !torch.vtensor<[1,1,2,?,4],f32>
  // CHECK: %[[LIST_5:[^ ]+]] = torch.prim.ListConstruct %[[UNSQUEEZE]], %[[UNSQUEEZE_1]] : (!torch.vtensor<[1,1,2,?,4],f32>, !torch.vtensor<[1,1,2,?,4],f32>) -> !torch.list<vtensor>
  // CHECK: %[[CAT_2:.*]] = torch.aten.cat %[[LIST_5]], %[[INT0]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[2,1,2,?,4],f32>
  // CHECK: return %[[FLATTEN]], %[[CAT_2]] : !torch.vtensor<[1,?,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>
  %none = torch.constant.none
  %0:2 = torch.operator "onnx.Attention"(%input, %weights, %bias, %none, %past) {torch.onnx.num_heads = 2 : si64} : (!torch.vtensor<[1,?,16],f32>, !torch.vtensor<[16,24],f32>, !torch.vtensor<[24],f32>, !torch.none, !torch.vtensor<[2,1,2,3,4],f32>) -> (!torch.vtensor<[1,?,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>)
  return %0#0, %0#1 : !torch.vtensor<[1,?,8],f32>, !torch.vtensor<[2,1,2,?,4],f32>
}

// -----

// Unidirectional (causal) masking with a dynamic sequence length. The causal
// mask shape is built from the runtime sequence length.

// CHECK-LABEL: func.func @test_com_microsoft_attention_unidirectional_dynamic_seq(
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: !torch.vtensor<[1,?,8],f32>
// CHECK-SAME: %[[WEIGHTS:[a-z0-9]+]]: !torch.vtensor<[8,24],f32>
// CHECK-SAME: %[[BIAS:[a-z0-9]+]]: !torch.vtensor<[24],f32>
func.func @test_com_microsoft_attention_unidirectional_dynamic_seq(%input: !torch.vtensor<[1,?,8],f32>, %weights: !torch.vtensor<[8,24],f32>, %bias: !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,?,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {ai.onnx = 17 : si64, com.microsoft = 1 : si64}} {
  // CHECK: %[[INT1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[INT0:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[INT2:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT3:[^ ]+]] = torch.constant.int 3
  // CHECK: %[[INT2_1:[^ ]+]] = torch.constant.int 2
  // CHECK: %[[INT4:[^ ]+]] = torch.constant.int 4
  // CHECK: %[[NONE:[^ ]+]] = torch.constant.none
  // CHECK: %[[MATMUL:.*]] = torch.aten.matmul %[[INPUT]], %[[WEIGHTS]] : !torch.vtensor<[1,?,8],f32>, !torch.vtensor<[8,24],f32> -> !torch.vtensor<[1,?,24],f32>
  // CHECK: %[[ADD:.*]] = torch.aten.add.Tensor %[[MATMUL]], %[[BIAS]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.vtensor<[24],f32>, !torch.int -> !torch.vtensor<[1,?,24],f32>
  // CHECK: %[[INT8:[^ ]+]] = torch.constant.int 8
  // CHECK: %[[INT16:[^ ]+]] = torch.constant.int 16
  // CHECK: %[[INT24:[^ ]+]] = torch.constant.int 24
  // CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT0]], %[[INT8]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[SLICE_1:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT8]], %[[INT16]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[SLICE_2:.*]] = torch.aten.slice.Tensor %[[ADD]], %[[INT2]], %[[INT16]], %[[INT24]], %[[INT1]] : !torch.vtensor<[1,?,24],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: %[[LIST:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN:.*]] = torch.aten.unflatten.int %[[SLICE]], %[[INT2]], %[[LIST]] : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[TRANSPOSE:.*]] = torch.aten.transpose.int %[[UNFLATTEN]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[LIST_1:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_1:.*]] = torch.aten.unflatten.int %[[SLICE_1]], %[[INT2]], %[[LIST_1]] : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[TRANSPOSE_1:.*]] = torch.aten.transpose.int %[[UNFLATTEN_1]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[LIST_2:[^ ]+]] = torch.prim.ListConstruct %[[INT2_1]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[UNFLATTEN_2:.*]] = torch.aten.unflatten.int %[[SLICE_2]], %[[INT2]], %[[LIST_2]] : !torch.vtensor<[1,?,8],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[TRANSPOSE_2:.*]] = torch.aten.transpose.int %[[UNFLATTEN_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[INT0_1:[^ ]+]] = torch.constant.int 0
  // CHECK: %[[SIZE:.*]] = torch.aten.size.int %[[INPUT]], %[[INT1]] : !torch.vtensor<[1,?,8],f32>, !torch.int -> !torch.int
  // CHECK: %[[ADD_1:.*]] = torch.aten.add.int %[[SIZE]], %[[INT0_1]] : !torch.int, !torch.int -> !torch.int
  // CHECK: %[[TRANSPOSE_3:.*]] = torch.aten.transpose.int %[[TRANSPOSE_1]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,2,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2,4,?],f32>
  // CHECK: %[[MATMUL_1:.*]] = torch.aten.matmul %[[TRANSPOSE]], %[[TRANSPOSE_3]] : !torch.vtensor<[1,2,?,4],f32>, !torch.vtensor<[1,2,4,?],f32> -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[FLOAT0_5:[^ ]+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[MUL:.*]] = torch.aten.mul.Scalar %[[MATMUL_1]], %[[FLOAT0_5]] : !torch.vtensor<[1,2,?,?],f32>, !torch.float -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[LIST_3:[^ ]+]] = torch.prim.ListConstruct %[[SIZE]], %[[ADD_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[FLOAT1:[^ ]+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[FULL:.*]] = torch.aten.full %[[LIST_3]], %[[FLOAT1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],f32>
  // CHECK: %[[INT1_1:[^ ]+]] = torch.constant.int 1
  // CHECK: %[[TRIU:.*]] = torch.aten.triu %[[FULL]], %[[INT1_1]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  // CHECK: %[[FLOATM10000:[^ ]+]] = torch.constant.float -1.000000e+04
  // CHECK: %[[MUL_1:.*]] = torch.aten.mul.Scalar %[[TRIU]], %[[FLOATM10000]] : !torch.vtensor<[?,?],f32>, !torch.float -> !torch.vtensor<[?,?],f32>
  // CHECK: %[[ADD_2:.*]] = torch.aten.add.Tensor %[[MUL]], %[[MUL_1]], %[[INT1]] : !torch.vtensor<[1,2,?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[SOFTMAX:.*]] = torch.aten.softmax.int %[[ADD_2]], %[[INT3]], %[[NONE]] : !torch.vtensor<[1,2,?,?],f32>, !torch.int, !torch.none -> !torch.vtensor<[1,2,?,?],f32>
  // CHECK: %[[MATMUL_2:.*]] = torch.aten.matmul %[[SOFTMAX]], %[[TRANSPOSE_2]] : !torch.vtensor<[1,2,?,?],f32>, !torch.vtensor<[1,2,?,4],f32> -> !torch.vtensor<[1,2,?,4],f32>
  // CHECK: %[[TRANSPOSE_4:.*]] = torch.aten.transpose.int %[[MATMUL_2]], %[[INT1]], %[[INT2]] : !torch.vtensor<[1,2,?,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,4],f32>
  // CHECK: %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[TRANSPOSE_4]], %[[INT2]], %[[INT3]] : !torch.vtensor<[1,?,2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8],f32>
  // CHECK: return %[[FLATTEN]] : !torch.vtensor<[1,?,8],f32>
  %0 = torch.operator "onnx.Attention"(%input, %weights, %bias) {torch.onnx.num_heads = 2 : si64, torch.onnx.unidirectional = 1 : si64} : (!torch.vtensor<[1,?,8],f32>, !torch.vtensor<[8,24],f32>, !torch.vtensor<[24],f32>) -> !torch.vtensor<[1,?,8],f32>
  return %0 : !torch.vtensor<[1,?,8],f32>
}
