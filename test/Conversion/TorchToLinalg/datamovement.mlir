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

// CHECK: #[[$INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:   func.func @torch.aten.reflection_pad2d(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,8,9],f32> {
// CHECK:           %[[CST0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,4],f32> -> tensor<1x1x4x4xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1x1x8x9xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_5]] : tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_4]][0, 0, 1, 1] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x4x4xf32> to tensor<1x1x2x2xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_4]][0, 0, 1, 1] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x4x4xf32> to tensor<1x1x2x2xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_4]][0, 0, 1, 1] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x4x4xf32> to tensor<1x1x2x2xf32>
// CHECK:           %[[VAL_10:.*]] = linalg.generic {indexing_maps = [#[[$INPUT_MAP]], #[[$INPUT_MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_9]] : tensor<1x1x2x2xf32>) outs(%[[VAL_8]] : tensor<1x1x2x2xf32>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_16:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_17:.*]] = arith.subi %[[VAL_3]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_3]], %[[VAL_15]] : index
// CHECK:             %[[VAL_19:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[CST0]], %[[CST0]], %[[VAL_18]], %[[VAL_17]]] : tensor<1x1x2x2xf32>
// CHECK:             linalg.yield %[[VAL_19]] : f32
// CHECK:           } -> tensor<1x1x2x2xf32>
// CHECK:           %[[VAL_20:.*]] = tensor.insert_slice %[[VAL_10]] into %[[VAL_6]][0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x2x2xf32> into tensor<1x1x8x9xf32>
// CHECK-COUNT-8:   linalg.generic
// CHECK:           %[[VAL_123:.*]] = tensor.insert_slice
// CHECK:           %[[VAL_124:.*]] = torch_c.from_builtin_tensor %[[VAL_123]] : tensor<1x1x8x9xf32> -> !torch.vtensor<[1,1,8,9],f32>
// CHECK:           return %[[VAL_124]] : !torch.vtensor<[1,1,8,9],f32>
// CHECK:         }

func.func @torch.aten.reflection_pad2d(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,8,9],f32>  {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3, %int2, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad2d %arg0, %0 : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int> -> !torch.vtensor<[1,1,8,9],f32>
  return %1 : !torch.vtensor<[1,1,8,9],f32>
}
