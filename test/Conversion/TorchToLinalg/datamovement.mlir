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

// -----

// CHECK-LABEL:   func.func @torch.aten.transpose.int$dynamic_dims(
// CHECK-SAME:                                                      %[[VAL_0:.*]]: !torch.vtensor<[1,56,56,96],f32>) -> !torch.vtensor<[?,?,?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,56,56,96],f32> -> tensor<1x56x56x96xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0], [1, 2], [3, 4], [5]] output_shape [1, 8, 7, 8, 7, 96] : tensor<1x56x56x96xf32> into tensor<1x8x7x8x7x96xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x8x8x7x7x96xf32>
// CHECK:           %[[TRANSPOSE:.*]] = linalg.transpose ins(%[[VAL_9]] {{.*}} outs(%[[EMPTY]] {{.*}} permutation = [0, 1, 3, 2, 4, 5]
// CHECK:           %[[RESULT_CAST:.*]] = tensor.cast %[[TRANSPOSE]] : tensor<1x8x8x7x7x96xf32> to tensor<?x?x?x?x?x?xf32>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[RESULT_CAST]] : tensor<?x?x?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?,?,?],f32>
// CHECK:           return %[[VAL_10]] : !torch.vtensor<[?,?,?,?,?,?],f32>
// CHECK:         }
func.func @torch.aten.transpose.int$dynamic_dims(%arg0: !torch.vtensor<[1,56,56,96],f32>) -> !torch.vtensor<[?,?,?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %int7 = torch.constant.int 7
  %int96 = torch.constant.int 96
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int1, %int8, %int7, %int8, %int7, %int96 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[1,56,56,96],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?,?,?],f32>
  %2 = torch.aten.transpose.int %1, %int2, %int3 : !torch.vtensor<[?,?,?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?,?,?,?],f32>
  return %2 : !torch.vtensor<[?,?,?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.unflatten.int$dynamic_input_dim(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[1,?,96],f32>) -> !torch.vtensor<[1,56,56,96],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,?,96],f32> -> tensor<1x?x96xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.cast %[[VAL_1]] : tensor<1x?x96xf32> to tensor<1x3136x96xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.expand_shape %[[VAL_2]] {{\[\[}}0], [1, 2], [3]] output_shape [1, 56, 56, 96] : tensor<1x3136x96xf32> into tensor<1x56x56x96xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<1x56x56x96xf32> -> !torch.vtensor<[1,56,56,96],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[1,56,56,96],f32>
// CHECK:         }
func.func @aten.unflatten.int$dynamic_input_dim(%arg0: !torch.vtensor<[1,?,96],f32>) -> !torch.vtensor<[1,56,56,96],f32> {
  %int1 = torch.constant.int 1
  %int56 = torch.constant.int 56
  %129 = torch.prim.ListConstruct %int56, %int56 : (!torch.int, !torch.int) -> !torch.list<int>
  %130 = torch.aten.unflatten.int %arg0, %int1, %129 : !torch.vtensor<[1,?,96],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[1,56,56,96],f32>
  return %130 : !torch.vtensor<[1,56,56,96],f32>
}

// -----

// CHECK-LABEL:   func.func @aten.permute$identity_permutation(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,32,16,8,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64,32,16,8,4],f32> -> tensor<64x32x16x8x4xf32>
// CHECK:           %[[VAL_2:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<64x32x16x8x4xf32> -> !torch.vtensor<[64,32,16,8,4],f32>
// CHECK:           return %[[VAL_2]] : !torch.vtensor<[64,32,16,8,4],f32>
// CHECK:         }
func.func @aten.permute$identity_permutation(%arg0: !torch.vtensor<[64,32,16,8,4],f32>) -> !torch.vtensor<[64,32,16,8,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3, %int4 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[64,32,16,8,4],f32>, !torch.list<int> -> !torch.vtensor<[64,32,16,8,4],f32>
  return %1 : !torch.vtensor<[64,32,16,8,4],f32>
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
