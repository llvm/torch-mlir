// RUN: npcomp-opt -torch-refine-types -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor {
// CHECK:           %[[SHAPED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[SHAPED]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
// CHECK:           %[[CASTED:.*]] = torch.tensor_static_info_cast %[[ARG]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[NONVAL_TENSOR:.*]] = torch.copy.tensor %[[CASTED]] : !torch.vtensor<[2,3,?],f32> -> !torch.tensor<[2,3,?],f32>
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[NONVAL_TENSOR]] : !torch.tensor<[2,3,?],f32> to !torch.tensor
// CHECK:           return %[[ERASED]] : !torch.tensor
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
  %1 = torch.copy.tensor %0 : !torch.vtensor -> !torch.tensor
  return %1 : !torch.tensor
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[ARG:.*]]: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor {
// CHECK:           %[[SHAPED:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[SHAPED]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor {
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[LHS:.*]]: !torch.vtensor<[2,?],f32>,
// CHECK-SAME:            %[[RHS:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
// CHECK:           %[[MM:.*]] = torch.aten.mm %[[LHS]], %[[RHS]] : !torch.vtensor<[2,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[MM]] : !torch.vtensor<[?,?],f32> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @f(%arg0: !torch.vtensor<[2,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
  %1 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func @f(
// CHECK-SAME:            %[[INPUT:.*]]: !torch.vtensor<[?,3],f32>,
// CHECK-SAME:            %[[WEIGHT:.*]]: !torch.vtensor<[5,3],f32>,
// CHECK-SAME:            %[[BIAS:.*]]: !torch.vtensor<[5],f32>) -> !torch.vtensor {
// CHECK:           %[[LINEAR:.*]] = torch.aten.linear %[[INPUT]], %[[WEIGHT]], %[[BIAS]] : !torch.vtensor<[?,3],f32>, !torch.vtensor<[5,3],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[?,?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[LINEAR]] : !torch.vtensor<[?,?],f32> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @f(%arg0: !torch.vtensor<[?,3],f32>, %arg1: !torch.vtensor<[5,3],f32>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor {
  %1 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[5,3],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL: func @f
// CHECK:           %[[CONV2D:.*]] = torch.aten.conv2d{{.*}} -> !torch.vtensor<[?,?,?,?],unk>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[CONV2D]] : !torch.vtensor<[?,?,?,?],unk> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @f(%arg0:!torch.vtensor, %arg1:!torch.vtensor, %arg2:!torch.vtensor) ->!torch.vtensor {
  %c0_i64 = torch.constant.int 0 : i64
  %c1_i64 = torch.constant.int 1 : i64
  %0 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  %1 = torch.prim.ListConstruct %c0_i64, %c0_i64 : (i64, i64) -> !torch.list<i64>
  %2 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  %3 = torch.aten.conv2d %arg0, %arg1, %arg2, %0, %1, %2, %c1_i64 : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.list<i64>, !torch.list<i64>, !torch.list<i64>, i64 ->!torch.vtensor
  return %3 :!torch.vtensor
}

// CHECK-LABEL: func @g
// CHECK:           %[[CONV2D:.*]] = torch.aten.conv2d{{.*}} -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[CONV2D]] : !torch.vtensor<[?,?,?,?],f32> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @g(%arg0:!torch.vtensor<*,f32>, %arg1:!torch.vtensor<*,f32>, %arg2:!torch.vtensor<*,f32>) ->!torch.vtensor {
  %c0_i64 = torch.constant.int 0 : i64
  %c1_i64 = torch.constant.int 1 : i64
  %0 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  %1 = torch.prim.ListConstruct %c0_i64, %c0_i64 : (i64, i64) -> !torch.list<i64>
  %2 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  %3 = torch.aten.conv2d %arg0, %arg1, %arg2, %0, %1, %2, %c1_i64 : !torch.vtensor<*,f32>, !torch.vtensor<*,f32>, !torch.vtensor<*,f32>, !torch.list<i64>, !torch.list<i64>, !torch.list<i64>, i64 ->!torch.vtensor
  return %3 :!torch.vtensor
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor {
  %c1_i64 = torch.constant.int 1 : i64
  %c3_i64 = torch.constant.int 3 : i64
  %c2_i64 = torch.constant.int 2 : i64
  %bool_false = basicpy.bool_constant false
  %21 = torch.prim.ListConstruct %c3_i64, %c3_i64 : (i64, i64) -> !torch.list<i64>
  %22 = torch.prim.ListConstruct %c2_i64, %c2_i64 : (i64, i64) -> !torch.list<i64>
  %23 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  %24 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  // CHECK: torch.aten.max_pool2d{{.*}} -> !torch.vtensor<[?,?,?,?],f32>
  %27 = torch.aten.max_pool2d %arg0, %21, %22, %23, %24, %bool_false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<i64>, !torch.list<i64>, !torch.list<i64>, !torch.list<i64>, !basicpy.BoolType -> !torch.vtensor
  return %27 : !torch.vtensor
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor {
  %c1_i64 = torch.constant.int 1 : i64
  %0 = torch.prim.ListConstruct %c1_i64, %c1_i64 : (i64, i64) -> !torch.list<i64>
  // CHECK: torch.aten.adaptive_avg_pool2d{{.*}} -> !torch.vtensor<[?,?,?,?],f32>
  %1 = torch.aten.adaptive_avg_pool2d %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<i64> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// Also test cast insertion for array types.
// CHECK-LABEL:   func @flatten_all(
// CHECK:           %[[FLATTENED:.*]] = torch.aten.flatten.using_ints{{.*}}-> !torch.tensor<[?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[FLATTENED]] : !torch.tensor<[?],f32> to !torch.tensor
// CHECK:           return %[[SHAPE_ERASED]]
func @flatten_all(%arg0: !torch.tensor<[3,2,?,5],f32>) -> !torch.tensor {
  %end = torch.constant.int -1 : i64
  %start = torch.constant.int 0 : i64
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !torch.tensor<[3,2,?,5],f32>, i64, i64 -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @flatten_some(
// CHECK:           torch.aten.flatten.using_ints{{.*}}-> !torch.tensor<[3,?,5],f32>
func @flatten_some(%arg0: !torch.tensor<[3,2,?,5],f32>) -> !torch.tensor {
  %end = torch.constant.int -2 : i64
  %start = torch.constant.int 1 : i64
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !torch.tensor<[3,2,?,5],f32>, i64, i64 -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @flatten_rank0(
// CHECK:           torch.aten.flatten.using_ints{{.*}}-> !torch.tensor<[?],f32>
func @flatten_rank0(%arg0: !torch.tensor<[],f32>) -> !torch.tensor {
  %end = torch.constant.int -1 : i64
  %start = torch.constant.int 0 : i64
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !torch.tensor<[],f32>, i64, i64 -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: !torch.vtensor<[4,6,3],f32>, %arg1: !torch.vtensor<[1,1,3],f32>, %arg2: !torch.vtensor<[?,3],f32>) {
  %c1_i64 = torch.constant.int 1 : i64
  // CHECK: torch.aten.add{{.*}} -> !torch.vtensor<[?,?,?],f32>
  %0 = torch.aten.add.Tensor %arg0, %arg1, %c1_i64 : !torch.vtensor<[4,6,3],f32>, !torch.vtensor<[1,1,3],f32>, i64 -> !torch.vtensor
  // CHECK: torch.aten.add{{.*}} -> !torch.vtensor<[?,?,?],f32>
  %1 = torch.aten.add.Tensor %arg0, %arg2, %c1_i64 : !torch.vtensor<[4,6,3],f32>, !torch.vtensor<[?,3],f32>, i64 -> !torch.vtensor
  return
}

// -----

// CHECK-LABEL:   func @f
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor {
  // Check propagation through multiple ops.
  // CHECK:           torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  // CHECK:           torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  // CHECK:           torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor
  %2 = torch.aten.tanh %1 : !torch.vtensor -> !torch.vtensor
  %3 = torch.aten.tanh %2 : !torch.vtensor -> !torch.vtensor
  return %3 : !torch.vtensor
}

// -----

// Check rewriting logic in case of mixes of users that do/don't allow type
// refinement.
// CHECK-LABEL:   func @f
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> (!torch.vtensor, !torch.vtensor) {
  // CHECK: %[[REFINED_TYPE:.*]] = torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  %1 = torch.aten.tanh %arg0 : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor
  // CHECK: %[[ORIGINAL_TYPE:.*]] = torch.tensor_static_info_cast %[[REFINED_TYPE]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
  // CHECK: torch.aten.tanh %[[REFINED_TYPE]] : !torch.vtensor<[2,3,?],f32> -> !torch.vtensor<[2,3,?],f32>
  %3 = torch.aten.tanh %1 : !torch.vtensor -> !torch.vtensor
  // CHECK: return %[[ORIGINAL_TYPE]], %[[ORIGINAL_TYPE]] : !torch.vtensor, !torch.vtensor
  return %1, %1 : !torch.vtensor, !torch.vtensor
}

// -----

// CHECK-LABEL:   func @f
// CHECK: %[[ATEN:.*]] = torch.aten.tanh %{{.*}} : !torch.vtensor -> !torch.vtensor<[2,3,?],f32>
// CHECK: %[[CAST:.*]] = torch.tensor_static_info_cast %[[ATEN]] : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
// CHECK: return %[[CAST]] : !torch.vtensor
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.vtensor {
  %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
  br ^bb1(%cast: !torch.vtensor)
^bb1(%arg1: !torch.vtensor):
  %1 = torch.aten.tanh %arg1 : !torch.vtensor -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func @f
// CHECK: func private @callee
// CHECK-NEXT: torch.aten.tanh %{{.*}} : !torch.vtensor -> !torch.vtensor<[2,3,?],f32>
func @f() {
  module {
    func private @callee(%arg0: !torch.vtensor) {
      %1 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
      return
    }
    func @caller(%arg0: !torch.vtensor<[2,3,?],f32>) {
      %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
      call @callee(%cast) : (!torch.vtensor) -> ()
      return
    }
  }
  return
}
