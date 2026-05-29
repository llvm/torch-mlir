// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @as_strided_static_rank2(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,2],f32> {
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[C3:.*]] = torch.constant.int 3
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C6:.*]] = torch.constant.int 6
// CHECK: %[[FLAT_SHAPE:.*]] = torch.prim.ListConstruct %[[C6]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STORAGE:.*]] = torch.aten.view %[[ARG0]], %[[FLAT_SHAPE]] : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[6],f32>
// CHECK: %[[RANGE0:.*]] = torch.aten.arange.start_step %[[C0]], %[[C2]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si64>
// CHECK: %[[SHAPE0:.*]] = torch.prim.ListConstruct %[[C2]], %[[C1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[AXIS0:.*]] = torch.aten.view %[[RANGE0]], %[[SHAPE0]] : !torch.vtensor<[2],si64>, !torch.list<int> -> !torch.vtensor<[2,1],si64>
// CHECK: %[[IDX0:.*]] = torch.aten.mul.Scalar %[[AXIS0]], %[[C3]] : !torch.vtensor<[2,1],si64>, !torch.int -> !torch.vtensor<[2,1],si64>
// CHECK: %[[RANGE1:.*]] = torch.aten.arange.start_step %[[C0]], %[[C2]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si64>
// CHECK: %[[SHAPE1:.*]] = torch.prim.ListConstruct %[[C1]], %[[C2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[AXIS1:.*]] = torch.aten.view %[[RANGE1]], %[[SHAPE1]] : !torch.vtensor<[2],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
// CHECK: %[[IDX1:.*]] = torch.aten.mul.Scalar %[[AXIS1]], %[[C1]] : !torch.vtensor<[1,2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
// CHECK: %[[GRID:.*]] = torch.aten.add.Tensor %[[IDX0]], %[[IDX1]], %[[C1]] : !torch.vtensor<[2,1],si64>, !torch.vtensor<[1,2],si64>, !torch.int -> !torch.vtensor<[2,2],si64>
// CHECK: %[[GRID_SHAPE:.*]] = torch.prim.ListConstruct %[[C4]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[FLAT_GRID:.*]] = torch.aten.view %[[GRID]], %[[GRID_SHAPE]] : !torch.vtensor<[2,2],si64>, !torch.list<int> -> !torch.vtensor<[4],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[FLAT_GRID]] : (!torch.vtensor<[4],si64>) -> !torch.list<vtensor>
// CHECK: %[[GATHERED:.*]] = torch.aten.index.Tensor_hacked_twin %[[STORAGE]], %[[INDEX_LIST]] : !torch.vtensor<[6],f32>, !torch.list<vtensor> -> !torch.vtensor<[4],f32>
// CHECK: %[[RESULT_SHAPE:.*]] = torch.prim.ListConstruct %[[C2]], %[[C2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESULT:.*]] = torch.aten.view %[[GATHERED]], %[[RESULT_SHAPE]] : !torch.vtensor<[4],f32>, !torch.list<int> -> !torch.vtensor<[2,2],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[2,2],f32>
func.func @as_strided_static_rank2(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,2],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %sizes = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %none : !torch.vtensor<[2,3],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2,2],f32>
  return %r : !torch.vtensor<[2,2],f32>
}

// -----

// CHECK-LABEL: func.func @decomposes_empty_rank2_without_index_grid(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[4],f32>) -> !torch.vtensor<[0,2],f32> {
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[EMPTY_IDX:.*]] = torch.vtensor.literal(dense<> : tensor<0xsi64>) : !torch.vtensor<[0],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[EMPTY_IDX]] : (!torch.vtensor<[0],si64>) -> !torch.list<vtensor>
// CHECK-NOT: torch.aten.add.Tensor
// CHECK: %[[GATHERED:.*]] = torch.aten.index.Tensor_hacked_twin %[[ARG0]], %[[INDEX_LIST]] : !torch.vtensor<[4],f32>, !torch.list<vtensor> -> !torch.vtensor<[0],f32>
// CHECK: %[[RESULT_SHAPE:.*]] = torch.prim.ListConstruct %[[C0]], %[[C2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[RESULT:.*]] = torch.aten.view %[[GATHERED]], %[[RESULT_SHAPE]] : !torch.vtensor<[0],f32>, !torch.list<int> -> !torch.vtensor<[0,2],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[0,2],f32>
func.func @decomposes_empty_rank2_without_index_grid(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[0,2],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %sizes = torch.prim.ListConstruct %int0, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %none : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[0,2],f32>
  return %r : !torch.vtensor<[0,2],f32>
}

// -----

// CHECK-LABEL: func.func @decomposes_empty_result_without_storage_bounds_check(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2],f32>) -> !torch.vtensor<[0],f32> {
// CHECK: %[[EMPTY_IDX:.*]] = torch.vtensor.literal(dense<> : tensor<0xsi64>) : !torch.vtensor<[0],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[EMPTY_IDX]] : (!torch.vtensor<[0],si64>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index.Tensor_hacked_twin %[[ARG0]], %[[INDEX_LIST]] : !torch.vtensor<[2],f32>, !torch.list<vtensor> -> !torch.vtensor<[0],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[0],f32>
func.func @decomposes_empty_result_without_storage_bounds_check(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[0],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %sizes = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %int3 : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[0],f32>
  return %r : !torch.vtensor<[0],f32>
}

// -----

// CHECK-LABEL: func.func @as_strided_rank0_input(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[FLAT_SHAPE:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STORAGE:.*]] = torch.aten.view %[[ARG0]], %[[FLAT_SHAPE]] : !torch.vtensor<[],f32>, !torch.list<int> -> !torch.vtensor<[1],f32>
// CHECK: %[[RANGE:.*]] = torch.aten.arange.start_step %[[C0]], %[[C1]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si64>
// CHECK: %[[IDX:.*]] = torch.aten.mul.Scalar %[[RANGE]], %[[C0]] : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[IDX]] : (!torch.vtensor<[1],si64>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index.Tensor_hacked_twin %[[STORAGE]], %[[INDEX_LIST]] : !torch.vtensor<[1],f32>, !torch.list<vtensor> -> !torch.vtensor<[1],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @as_strided_rank0_input(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %sizes = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %none : !torch.vtensor<[],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[1],f32>
  return %r : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL: func.func @as_strided_rank0_result(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[4],f32>) -> !torch.vtensor<[],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[RANGE:.*]] = torch.aten.arange.start_step %[[C0]], %[[C1]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[RANGE]] : (!torch.vtensor<[1],si64>) -> !torch.list<vtensor>
// CHECK: %[[GATHERED:.*]] = torch.aten.index.Tensor_hacked_twin %[[ARG0]], %[[INDEX_LIST]] : !torch.vtensor<[4],f32>, !torch.list<vtensor> -> !torch.vtensor<[1],f32>
// CHECK: %[[SCALAR_SHAPE:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK: %[[RESULT:.*]] = torch.aten.view %[[GATHERED]], %[[SCALAR_SHAPE]] : !torch.vtensor<[1],f32>, !torch.list<int> -> !torch.vtensor<[],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[],f32>
func.func @as_strided_rank0_result(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %sizes = torch.prim.ListConstruct : () -> !torch.list<int>
  %strides = torch.prim.ListConstruct : () -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %none : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[],f32>
  return %r : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @as_strided_after_slice_default_offset(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C12:.*]] = torch.constant.int 12
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[FLAT_SHAPE:.*]] = torch.prim.ListConstruct %[[C12]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STORAGE:.*]] = torch.aten.view %[[ARG0]], %[[FLAT_SHAPE]] : !torch.vtensor<[3,4],f32>, !torch.list<int> -> !torch.vtensor<[12],f32>
// CHECK: %[[RANGE:.*]] = torch.aten.arange.start_step %[[C0]], %[[C2]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si64>
// CHECK: %[[SCALED:.*]] = torch.aten.mul.Scalar %[[RANGE]], %[[C4]] : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[2],si64>
// CHECK: %[[OFFSET_IDX:.*]] = torch.aten.add.Scalar %[[SCALED]], %[[C4]], %[[C1]] : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[OFFSET_IDX]] : (!torch.vtensor<[2],si64>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index.Tensor_hacked_twin %[[STORAGE]], %[[INDEX_LIST]] : !torch.vtensor<[12],f32>, !torch.list<vtensor> -> !torch.vtensor<[2],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @as_strided_after_slice_default_offset(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int1, %int3, %int1 : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
  %r = torch.aten.as_strided %slice, %sizes, %strides, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @as_strided_after_slice_explicit_zero(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C12:.*]] = torch.constant.int 12
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[FLAT_SHAPE:.*]] = torch.prim.ListConstruct %[[C12]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STORAGE:.*]] = torch.aten.view %[[ARG0]], %[[FLAT_SHAPE]] : !torch.vtensor<[3,4],f32>, !torch.list<int> -> !torch.vtensor<[12],f32>
// CHECK: %[[RANGE:.*]] = torch.aten.arange.start_step %[[C0]], %[[C2]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2],si64>
// CHECK: %[[SCALED:.*]] = torch.aten.mul.Scalar %[[RANGE]], %[[C4]] : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[2],si64>
// CHECK-NOT: torch.aten.add.Scalar
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[SCALED]] : (!torch.vtensor<[2],si64>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index.Tensor_hacked_twin %[[STORAGE]], %[[INDEX_LIST]] : !torch.vtensor<[12],f32>, !torch.list<vtensor> -> !torch.vtensor<[2],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @as_strided_after_slice_explicit_zero(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int1, %int3, %int1 : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
  %r = torch.aten.as_strided %slice, %sizes, %strides, %int0 : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_dynamic_slice_with_nonzero_start(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[C3:.*]] = torch.constant.int 3
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C4]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[C0]], %[[C1]], %[[C3]], %[[C1]] : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[SLICE]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[?,4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_dynamic_slice_with_nonzero_start(%arg0: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int1, %int3, %int1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  %r = torch.aten.as_strided %slice, %sizes, %strides, %none : !torch.vtensor<[?,4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_nonempty_dynamic_storage_size(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C4]] : (!torch.int) -> !torch.list<int>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[ARG0]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[?,4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_nonempty_dynamic_storage_size(%arg0: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int0, %int3, %int1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  %r = torch.aten.as_strided %slice, %sizes, %strides, %none : !torch.vtensor<[?,4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @empty_diagonal_keeps_base_offset(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[1],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C6:.*]] = torch.constant.int 6
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[FLAT_SHAPE:.*]] = torch.prim.ListConstruct %[[C6]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STORAGE:.*]] = torch.aten.view %[[ARG0]], %[[FLAT_SHAPE]] : !torch.vtensor<[2,3],f32>, !torch.list<int> -> !torch.vtensor<[6],f32>
// CHECK: %[[RANGE:.*]] = torch.aten.arange.start_step %[[C0]], %[[C1]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si64>
// CHECK: %[[IDX:.*]] = torch.aten.mul.Scalar %[[RANGE]], %[[C1]] : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
// CHECK-NOT: torch.aten.add.Scalar
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[IDX]] : (!torch.vtensor<[1],si64>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index.Tensor_hacked_twin %[[STORAGE]], %[[INDEX_LIST]] : !torch.vtensor<[6],f32>, !torch.list<vtensor> -> !torch.vtensor<[1],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @empty_diagonal_keeps_base_offset(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %sizes = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %diag = torch.aten.diagonal %arg0, %int3, %int0, %int1 : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[0],f32>
  %r = torch.aten.as_strided %diag, %sizes, %strides, %none : !torch.vtensor<[0],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[1],f32>
  return %r : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_nonconstant_storage_offset(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[4],f32>, %[[OFFSET:.*]]: !torch.int) -> !torch.vtensor<[2],f32> {
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[ARG0]], %[[SIZES]], %[[STRIDES]], %[[OFFSET]] : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_nonconstant_storage_offset(%arg0: !torch.vtensor<[4],f32>, %offset: !torch.int) -> !torch.vtensor<[2],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %offset : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_negative_stride(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[CN1:.*]] = torch.constant.int -1
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[CN1]] : (!torch.int) -> !torch.list<int>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[ARG0]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_negative_stride(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %intm1 = torch.constant.int -1
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %intm1 : (!torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %none : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_static_out_of_bounds_access(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[C3:.*]] = torch.constant.int 3
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[ARG0]], %[[SIZES]], %[[STRIDES]], %[[C3]] : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_static_out_of_bounds_access(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %r = torch.aten.as_strided %arg0, %sizes, %strides, %int3 : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_dynamic_view_after_slice_offset(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[CN1:.*]] = torch.constant.int -1
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[C3:.*]] = torch.constant.int 3
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C4]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[VIEW_SHAPE:.*]] = torch.prim.ListConstruct %[[CN1]], %[[C2]], %[[C2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[C0]], %[[C1]], %[[C3]], %[[C1]] : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
// CHECK: %[[VIEW:.*]] = torch.aten.view %[[SLICE]], %[[VIEW_SHAPE]] : !torch.vtensor<[?,4],f32>, !torch.list<int> -> !torch.vtensor<[?,2,2],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[VIEW]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[?,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_dynamic_view_after_slice_offset(%arg0: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %intm1 = torch.constant.int -1
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %view_shape = torch.prim.ListConstruct %intm1, %int2, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int1, %int3, %int1 : !torch.vtensor<[?,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,4],f32>
  %view = torch.aten.view %slice, %view_shape : !torch.vtensor<[?,4],f32>, !torch.list<int> -> !torch.vtensor<[?,2,2],f32>
  %r = torch.aten.as_strided %view, %sizes, %strides, %none : !torch.vtensor<[?,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_refined_static_out_of_bounds_access(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[3],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C3:.*]] = torch.constant.int 3
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C3]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[CAST:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[?],f32> to !torch.vtensor<[2],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[CAST]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[3],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[3],f32>
func.func @does_not_decompose_refined_static_out_of_bounds_access(%arg0: !torch.vtensor<[?],f32>) -> !torch.vtensor<[3],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %sizes = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[?],f32> to !torch.vtensor<[2],f32>
  %r = torch.aten.as_strided %cast, %sizes, %strides, %none : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[3],f32>
  return %r : !torch.vtensor<[3],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_after_real_view(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2],complex<f32>>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[REAL:.*]] = torch.aten.real %[[ARG0]] : !torch.vtensor<[2],complex<f32>> -> !torch.vtensor<[2],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[REAL]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_after_real_view(%arg0: !torch.vtensor<[2],complex<f32>>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %real = torch.aten.real %arg0 : !torch.vtensor<[2],complex<f32>> -> !torch.vtensor<[2],f32>
  %r = torch.aten.as_strided %real, %sizes, %strides, %none : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_after_imag_view(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2],complex<f32>>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[IMAG:.*]] = torch.aten.imag %[[ARG0]] : !torch.vtensor<[2],complex<f32>> -> !torch.vtensor<[2],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[IMAG]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_after_imag_view(%arg0: !torch.vtensor<[2],complex<f32>>) -> !torch.vtensor<[2],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %imag = torch.aten.imag %arg0 : !torch.vtensor<[2],complex<f32>> -> !torch.vtensor<[2],f32>
  %r = torch.aten.as_strided %imag, %sizes, %strides, %none : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_after_contiguous(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C2:.*]] = torch.constant.int 2
// CHECK: %[[C3:.*]] = torch.constant.int 3
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C2]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C4]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[SLICE:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[C0]], %[[C1]], %[[C3]], %[[C1]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK: %[[CONTIG:.*]] = torch.aten.contiguous %[[SLICE]], %[[C0]] : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[CONTIG]], %[[SIZES]], %[[STRIDES]], %[[C0]] : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @does_not_decompose_after_contiguous(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %sizes = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int1, %int3, %int1 : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,4],f32>
  %contiguous = torch.aten.contiguous %slice, %int0 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  %r = torch.aten.as_strided %contiguous, %sizes, %strides, %int0 : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2],f32>
  return %r : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_oob_after_expand_leading_singleton_slice(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1],f32> {
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[C4:.*]] = torch.constant.int 4
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[ARG0]], %[[SIZES]], %[[STRIDES]], %[[C4]] : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @does_not_decompose_oob_after_expand_leading_singleton_slice(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %expand_shape = torch.prim.ListConstruct %int1, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %sizes = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %expanded = torch.aten.expand %arg0, %expand_shape, %false : !torch.vtensor<[4],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,4],f32>
  %slice = torch.aten.slice.Tensor %expanded, %int0, %int1, %int1, %int1 : !torch.vtensor<[1,4],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[0,4],f32>
  %r = torch.aten.as_strided %slice, %sizes, %strides, %none : !torch.vtensor<[0,4],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[1],f32>
  return %r : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL: func.func @uses_select_stride_for_zero_numel_view(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[6],f32>) -> !torch.vtensor<[1],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[RANGE:.*]] = torch.aten.arange.start_step %[[C0]], %[[C1]], %[[C1]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si64>
// CHECK: %[[SCALED:.*]] = torch.aten.mul.Scalar %[[RANGE]], %[[C1]] : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
// CHECK: %[[OFFSET_IDX:.*]] = torch.aten.add.Scalar %[[SCALED]], %[[C1]], %[[C1]] : !torch.vtensor<[1],si64>, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
// CHECK: %[[INDEX_LIST:.*]] = torch.prim.ListConstruct %[[OFFSET_IDX]] : (!torch.vtensor<[1],si64>) -> !torch.list<vtensor>
// CHECK: %[[RESULT:.*]] = torch.aten.index.Tensor_hacked_twin %[[ARG0]], %[[INDEX_LIST]] : !torch.vtensor<[6],f32>, !torch.list<vtensor> -> !torch.vtensor<[1],f32>
// CHECK-NOT: torch.aten.as_strided
// CHECK: return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @uses_select_stride_for_zero_numel_view(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %sizes = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %view_shape = torch.prim.ListConstruct %int2, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %slice = torch.aten.slice.Tensor %arg0, %int0, %int0, %int0, %int1 : !torch.vtensor<[6],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[0],f32>
  %view = torch.aten.view %slice, %view_shape : !torch.vtensor<[0],f32>, !torch.list<int> -> !torch.vtensor<[2,0],f32>
  %select = torch.aten.select.int %view, %int0, %int1 : !torch.vtensor<[2,0],f32>, !torch.int, !torch.int -> !torch.vtensor<[0],f32>
  %r = torch.aten.as_strided %select, %sizes, %strides, %none : !torch.vtensor<[0],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[1],f32>
  return %r : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL: func.func @does_not_decompose_diagonal_intmin_offset(
// CHECK-SAME: %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[1],f32> {
// CHECK: %[[NONE:.*]] = torch.constant.none
// CHECK: %[[C0:.*]] = torch.constant.int 0
// CHECK: %[[C1:.*]] = torch.constant.int 1
// CHECK: %[[CMIN:.*]] = torch.constant.int -9223372036854775808
// CHECK: %[[SIZES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[STRIDES:.*]] = torch.prim.ListConstruct %[[C1]] : (!torch.int) -> !torch.list<int>
// CHECK: %[[DIAG:.*]] = torch.aten.diagonal %[[ARG0]], %[[CMIN]], %[[C0]], %[[C1]] : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
// CHECK-NOT: torch.aten.index.Tensor_hacked_twin
// CHECK: %[[RESULT:.*]] = torch.aten.as_strided %[[DIAG]], %[[SIZES]], %[[STRIDES]], %[[NONE]] : !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[1],f32>
// CHECK: return %[[RESULT]] : !torch.vtensor<[1],f32>
func.func @does_not_decompose_diagonal_intmin_offset(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[1],f32> {
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %intmin = torch.constant.int -9223372036854775808
  %sizes = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %strides = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %diag = torch.aten.diagonal %arg0, %intmin, %int0, %int1 : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],f32>
  %r = torch.aten.as_strided %diag, %sizes, %strides, %none : !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.none -> !torch.vtensor<[1],f32>
  return %r : !torch.vtensor<[1],f32>
}
