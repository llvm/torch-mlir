// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.permute(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,8,4,32,16],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64,32,16,8,4],f32> -> tensor<64x32x16x8x4xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x8x4x32x16xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.transpose ins(%[[VAL_1]] : tensor<64x32x16x8x4xf32>) outs(%[[VAL_2]] : tensor<64x8x4x32x16xf32>) permutation = [0, 3, 4, 1, 2]
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<64x8x4x32x16xf32> -> !torch.vtensor<[64,8,4,32,16],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[64,8,4,32,16],f32>
// CHECK:         }
func.func @torch.aten.permute(%arg0: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,8,4,32,16],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int0, %int3, %int4, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[64,32,16,8,4],f32>, !torch.list<int> -> !torch.vtensor<[64,8,4,32,16],f32>
  return %1 : !torch.vtensor<[64,8,4,32,16],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.permute$rank0(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:           %[[VAL_2:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_2]] : !torch.vtensor<[],f32>
// CHECK:         }
func.func @torch.aten.permute$rank0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
  return %1 : !torch.vtensor<[],f32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4 * 2 + d5 + d1 * 4, d3 + d2 * 16)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>
// CHECK: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
// CHECK-LABEL: func.func @torch.aten.col2im(
// CHECK-SAME:                                %[[VAL_ARG0:.*]]: !torch.vtensor<[1,12,128],f32>) -> !torch.vtensor<[1,3,14,30],f32> {
// CHECK: %[[VAL_CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<1x3x16x32xf32>
// CHECK: %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_ARG0]] : !torch.vtensor<[1,12,128],f32> -> tensor<1x12x128xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<2x2xf32>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<8x16xf32>
// CHECK: %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_CST:.*]] : f32) outs(%[[VAL_0]] : tensor<1x3x16x32xf32>) -> tensor<1x3x16x32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<1x12x128xf32>, tensor<2x2xf32>, tensor<8x16xf32>) outs(%[[VAL_4]] : tensor<1x3x16x32xf32>) {
// CHECK: ^bb0(%[[VAL_IN0:.*]]: f32, %[[VAL_IN1:.*]]: f32, %[[VAL_IN2:.*]]: f32, %[[VAL_OUT:.*]]: f32):
// CHECK:   %[[VAL_7:.*]] = arith.addf %[[VAL_IN0]], %[[VAL_OUT]] : f32
// CHECK:   linalg.yield %[[VAL_7]] : f32
// CHECK: } -> tensor<1x3x16x32xf32>
// CHECK: %[[VAL_SLICE:.*]] = tensor.extract_slice %[[VAL_5]][0, 0, 1, 1] [1, 3, 14, 30] [1, 1, 1, 1] : tensor<1x3x16x32xf32> to tensor<1x3x14x30xf32>
// CHECK: %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_SLICE]] : tensor<1x3x14x30xf32> -> !torch.vtensor<[1,3,14,30],f32>
// CHECK: return %[[VAL_6]] : !torch.vtensor<[1,3,14,30],f32>
func.func @torch.aten.col2im(%arg0: !torch.vtensor<[1,12,128],f32>) -> !torch.vtensor<[1,3,14,30],f32> {
  %int14 = torch.constant.int 14
  %int30 = torch.constant.int 30
  %0 = torch.prim.ListConstruct %int14, %int30 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %int2_0 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %int2, %int2_0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %int1_1 = torch.constant.int 1
  %2 = torch.prim.ListConstruct %int1, %int1_1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1_2 = torch.constant.int 1
  %int1_3 = torch.constant.int 1
  %3 = torch.prim.ListConstruct %int1_2, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
  %int2_4 = torch.constant.int 2
  %int2_5 = torch.constant.int 2
  %4 = torch.prim.ListConstruct %int2_4, %int2_5 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.aten.col2im %arg0, %0, %1, %2, %3, %4 : !torch.vtensor<[1,12,128],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[1,3,14,30],f32>
  return %5 : !torch.vtensor<[1,3,14,30],f32>
}
