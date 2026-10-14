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

// Every reflected tile reads the whole input at reflected indices with an
// identity indexing map; no tile reads a slice of the input.
// CHECK: #[[$INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NOT:       -d{{[0-9]}}
// CHECK-LABEL:   func.func @torch.aten.reflection_pad2d(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,8,9],f32> {
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,1,4,4],f32> -> tensor<1x1x4x4xf32>
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<1x1x2x2xf32>
// CHECK:           %[[TILE:.*]] = linalg.generic {indexing_maps = [#[[$INPUT_MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[INIT]] : tensor<1x1x2x2xf32>) {
// CHECK:             %[[ROW:.*]] = linalg.index 2 : index
// CHECK:             %[[COL:.*]] = linalg.index 3 : index
// CHECK:             %[[RCOL:.*]] = arith.subi %{{.*}}, %[[COL]] : index
// CHECK:             %[[RROW:.*]] = arith.subi %{{.*}}, %[[ROW]] : index
// CHECK:             %[[V:.*]] = tensor.extract %[[INPUT]]{{\[}}%{{.*}}, %{{.*}}, %[[RROW]], %[[RCOL]]] : tensor<1x1x4x4xf32>
// CHECK:             linalg.yield %[[V]] : f32
// CHECK:           } -> tensor<1x1x2x2xf32>
// CHECK:           tensor.insert_slice %[[TILE]] into %{{.*}}[0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x2x2xf32> into tensor<1x1x8x9xf32>
// CHECK-COUNT-7:   tensor.extract %[[INPUT]]
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[LAST:.*]] = tensor.insert_slice %[[INPUT]] into %{{.*}}[0, 0, 2, 2] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x1x4x4xf32> into tensor<1x1x8x9xf32>
// CHECK:           %[[RES:.*]] = torch_c.from_builtin_tensor %[[LAST]] : tensor<1x1x8x9xf32> -> !torch.vtensor<[1,1,8,9],f32>
// CHECK:           return %[[RES]] : !torch.vtensor<[1,1,8,9],f32>
// CHECK:         }

func.func @torch.aten.reflection_pad2d(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,8,9],f32>  {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.prim.ListConstruct %int2, %int3, %int2, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad2d %arg0, %0 : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int> -> !torch.vtensor<[1,1,8,9],f32>
  return %1 : !torch.vtensor<[1,1,8,9],f32>
}

// -----

// The reflected tiles must use identity indexing maps: a reflected map such as
// (d0, d1, d2) -> (d0, d1, -d2 + 2) makes linalg tiling assert. They must also
// read the whole input at absolute columns rather than a slice of it: the
// vectorizer lowers the tensor.extract to a vector.gather whose index is
// linearized as if its source were contiguous, which a multi-row slice is not.
// CHECK: #[[$INPUT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NOT:     -d{{[0-9]}}
// CHECK-LABEL:   func.func @torch.aten.reflection_pad1d(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[1,2,4],f32>) -> !torch.vtensor<[1,2,8],f32> {
// CHECK-DAG:       %[[CST0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[CST2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[CST3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,2,4],f32> -> tensor<1x2x4xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.fill
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[LEFT_INIT:.*]] = tensor.empty() : tensor<1x2x3xf32>
// CHECK:           %[[LEFT:.*]] = linalg.generic {indexing_maps = [#[[$INPUT_MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} outs(%[[LEFT_INIT]] : tensor<1x2x3xf32>) {
// CHECK:             %[[IDX1:.*]] = linalg.index 1 : index
// CHECK:             %[[IDX2:.*]] = linalg.index 2 : index
// CHECK:             %[[COL:.*]] = arith.subi %[[CST3]], %[[IDX2]] : index
// CHECK:             %[[VAL_3:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[CST0]], %[[IDX1]], %[[COL]]] : tensor<1x2x4xf32>
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<1x2x3xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.insert_slice %[[LEFT]] into %[[VAL_2]][0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<1x2x3xf32> into tensor<1x2x8xf32>
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[RIGHT:.*]] = linalg.generic {indexing_maps = [#[[$INPUT_MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} outs(%{{.*}} : tensor<1x2x1xf32>) {
// CHECK:             tensor.extract %[[VAL_1]]{{\[}}%[[CST0]], %{{.*}}, %[[CST2]]] : tensor<1x2x4xf32>
// CHECK:           } -> tensor<1x2x1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.insert_slice %[[RIGHT]] into %[[VAL_4]][0, 0, 7] [1, 2, 1] [1, 1, 1] : tensor<1x2x1xf32> into tensor<1x2x8xf32>
// CHECK-NOT:       tensor.extract_slice
// CHECK:           %[[VAL_6:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_5]][0, 0, 3] [1, 2, 4] [1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x8xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<1x2x8xf32> -> !torch.vtensor<[1,2,8],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[1,2,8],f32>
// CHECK:         }

func.func @torch.aten.reflection_pad1d(%arg0: !torch.vtensor<[1,2,4],f32>) -> !torch.vtensor<[1,2,8],f32> {
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[1,2,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,8],f32>
  return %1 : !torch.vtensor<[1,2,8],f32>
}

// -----

// Dynamic width: each pad must be smaller than the width, checked at runtime.
// CHECK-LABEL:   func.func @torch.aten.reflection_pad1d$dynamic(
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[?,2,?],f32> -> tensor<?x2x?xf32>
// CHECK:           %[[W:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<?x2x?xf32>
// CHECK:           %[[OK_L:.*]] = arith.cmpi sgt, %[[W]], %[[C2]] : index
// CHECK:           cf.assert %[[OK_L]], "padding must be smaller than the input width"
// CHECK:           %[[OK_R:.*]] = arith.cmpi sgt, %[[W]], %[[C1]] : index
// CHECK:           cf.assert %[[OK_R]], "padding must be smaller than the input width"
// CHECK:           %[[RCOL:.*]] = arith.subi %[[W]], %[[C2]] : index
// CHECK:           linalg.generic {{.*}} outs(%{{.*}} : tensor<?x2x2xf32>)
// CHECK:             tensor.extract %[[INPUT]]
// CHECK:           linalg.generic {{.*}} outs(%{{.*}} : tensor<?x2x1xf32>)
// CHECK:             tensor.extract %[[INPUT]]{{\[}}%{{.*}}, %{{.*}}, %[[RCOL]]] : tensor<?x2x?xf32>
func.func @torch.aten.reflection_pad1d$dynamic(%arg0: !torch.vtensor<[?,2,?],f32>) -> !torch.vtensor<[?,2,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[?,2,?],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?],f32>
  return %1 : !torch.vtensor<[?,2,?],f32>
}

// -----

// A pad as wide as the input would read past it.
func.func @torch.aten.reflection_pad1d$pad_too_large(%arg0: !torch.vtensor<[1,2,4],f32>) -> !torch.vtensor<[1,2,9],f32> {
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int4, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error@+1 {{failed to legalize operation 'torch.aten.reflection_pad1d'}}
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[1,2,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,9],f32>
  return %1 : !torch.vtensor<[1,2,9],f32>
}

// -----

func.func @torch.aten.reflection_pad1d$negative_pad(%arg0: !torch.vtensor<[1,2,4],f32>) -> !torch.vtensor<[1,2,4],f32> {
  %int-1 = torch.constant.int -1
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int-1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error@+1 {{failed to legalize operation 'torch.aten.reflection_pad1d'}}
  %1 = torch.aten.reflection_pad1d %arg0, %0 : !torch.vtensor<[1,2,4],f32>, !torch.list<int> -> !torch.vtensor<[1,2,4],f32>
  return %1 : !torch.vtensor<[1,2,4],f32>
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

// CHECK-LABEL:   func.func @torch.aten.slice$end_int64_max_dynamic(
// CHECK-SAME:                  %[[ARG0:.*]]: !torch.vtensor<[4,?],f32>
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[T:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,?],f32> -> tensor<4x?xf32>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[T]], %[[C1]] : tensor<4x?xf32>
// COM: No INT64_MAX sentinel is materialized: the end resolves to %[[DIM]], so
// COM: the slice extent is computed as (dim - start) directly off the queried
// COM: dim, with no min(INT64_MAX, dim) clamp in between.
// CHECK-NOT:       9223372036854775807
// CHECK:           arith.subi %[[DIM]], %{{.*}} : index
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %[[T]][0, %{{.*}}] [4, %{{.*}}] [1, 1] : tensor<4x?xf32> to tensor<4x?xf32>
// CHECK:           torch_c.from_builtin_tensor %[[SLICE]]
func.func @torch.aten.slice$end_int64_max_dynamic(%arg0: !torch.vtensor<[4,?],f32>) -> !torch.vtensor<[4,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %intmax = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int2, %intmax, %int1 : !torch.vtensor<[4,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?],f32>
  return %0 : !torch.vtensor<[4,?],f32>
}

// -----

// Same INT64_MAX end sentinel, but with step=2: the end still resolves to the
// dim (no sentinel constant), and the strided size is (dim - start) floor-div
// step.
// CHECK-LABEL:   func.func @torch.aten.slice$end_int64_max_dynamic_step2(
// CHECK-SAME:                  %[[ARG0:.*]]: !torch.vtensor<[4,?],f32>
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[T:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,?],f32> -> tensor<4x?xf32>
// CHECK:           %[[DIM:.*]] = tensor.dim %[[T]], %[[C1]] : tensor<4x?xf32>
// CHECK-NOT:       9223372036854775807
// CHECK:           %[[LEN:.*]] = arith.subi %[[DIM]], %{{.*}} : index
// CHECK:           arith.floordivsi %{{.*}}, %{{.*}} : index
// CHECK:           tensor.extract_slice %[[T]][0, %{{.*}}] [4, %{{.*}}] [1, 2] : tensor<4x?xf32> to tensor<4x?xf32>
func.func @torch.aten.slice$end_int64_max_dynamic_step2(%arg0: !torch.vtensor<[4,?],f32>) -> !torch.vtensor<[4,?],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %intmax = torch.constant.int 9223372036854775807
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int2, %intmax, %int2 : !torch.vtensor<[4,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?],f32>
  return %0 : !torch.vtensor<[4,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.slice.Tensor$negative_step(
// CHECK-SAME:                                                     %[[ARG0:.*]]: !torch.vtensor<[10],f32>) -> !torch.vtensor<[9],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[10],f32> -> tensor<10xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic
// CHECK:           %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0] [9] [1] : tensor<10xf32> to tensor<9xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<9xf32> -> !torch.vtensor<[9],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[9],f32>
// CHECK:         }
func.func @torch.aten.slice.Tensor$negative_step(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[9],f32> {
  %int0 = torch.constant.int 0
  %int9 = torch.constant.int 9
  %int_neg1 = torch.constant.int -1
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int9, %int0, %int_neg1 : !torch.vtensor<[10],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[9],f32>
  return %0 : !torch.vtensor<[9],f32>
}

// -----

// Verify that diag_embed with non-zero offset clamps the extract index
// to prevent out-of-bounds access on the input tensor.

// CHECK-LABEL:   func.func @torch.aten.diag_embed$positive_offset(
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[CLAMPED:.*]] = arith.minui %{{.*}}, %[[C2]] : index
// CHECK:           %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%[[CLAMPED]]] : tensor<3xf32>
func.func @torch.aten.diag_embed$positive_offset(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[4,4],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %int_neg1 = torch.constant.int -1
  %0 = torch.aten.diag_embed %arg0, %int1, %int0, %int_neg1 : !torch.vtensor<[3],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}
