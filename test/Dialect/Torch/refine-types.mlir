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
// CHECK:           %[[NONVAL_TENSOR:.*]] = torch.copy.to_tensor %[[CASTED]] : !torch.tensor<[2,3,?],f32>
// CHECK:           %[[ERASED:.*]] = torch.tensor_static_info_cast %[[NONVAL_TENSOR]] : !torch.tensor<[2,3,?],f32> to !torch.tensor
// CHECK:           return %[[ERASED]] : !torch.tensor
func @f(%arg0: !torch.vtensor<[2,3,?],f32>) -> !torch.tensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2,3,?],f32> to !torch.vtensor
  %1 = torch.copy.to_tensor %0 : !torch.tensor
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
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %3 = torch.aten.conv2d %arg0, %arg1, %arg2, %0, %1, %2, %int1 : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int ->!torch.vtensor
  return %3 :!torch.vtensor
}

// CHECK-LABEL: func @g
// CHECK:           %[[CONV2D:.*]] = torch.aten.conv2d{{.*}} -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[CONV2D]] : !torch.vtensor<[?,?,?,?],f32> to !torch.vtensor
// CHECK:           return %[[SHAPE_ERASED]] : !torch.vtensor
func @g(%arg0:!torch.vtensor<*,f32>, %arg1:!torch.vtensor<*,f32>, %arg2:!torch.vtensor<*,f32>) ->!torch.vtensor {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %3 = torch.aten.conv2d %arg0, %arg1, %arg2, %0, %1, %2, %int1 : !torch.vtensor<*,f32>, !torch.vtensor<*,f32>, !torch.vtensor<*,f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.int ->!torch.vtensor
  return %3 :!torch.vtensor
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor {
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %bool_false = torch.constant.bool false
  %21 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %22 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %23 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %24 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  // CHECK: torch.aten.max_pool2d{{.*}} -> !torch.vtensor<[?,?,?,?],f32>
  %27 = torch.aten.max_pool2d %arg0, %21, %22, %23, %24, %bool_false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.bool -> !torch.vtensor
  return %27 : !torch.vtensor
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  // CHECK: torch.aten.adaptive_avg_pool2d{{.*}} -> !torch.vtensor<[?,?,?,?],f32>
  %1 = torch.aten.adaptive_avg_pool2d %arg0, %0 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<!torch.int> -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// Also test cast insertion for array types.
// CHECK-LABEL:   func @flatten_all(
// CHECK:           %[[FLATTENED:.*]] = torch.aten.flatten.using_ints{{.*}}-> !torch.tensor<[?],f32>
// CHECK:           %[[SHAPE_ERASED:.*]] = torch.tensor_static_info_cast %[[FLATTENED]] : !torch.tensor<[?],f32> to !torch.tensor
// CHECK:           return %[[SHAPE_ERASED]]
func @flatten_all(%arg0: !torch.tensor<[3,2,?,5],f32>) -> !torch.tensor {
  %end = torch.constant.int -1
  %start = torch.constant.int 0
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !torch.tensor<[3,2,?,5],f32>, !torch.int, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @flatten_some(
// CHECK:           torch.aten.flatten.using_ints{{.*}}-> !torch.tensor<[3,?,5],f32>
func @flatten_some(%arg0: !torch.tensor<[3,2,?,5],f32>) -> !torch.tensor {
  %end = torch.constant.int -2
  %start = torch.constant.int 1
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !torch.tensor<[3,2,?,5],f32>, !torch.int, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @flatten_rank0(
// CHECK:           torch.aten.flatten.using_ints{{.*}}-> !torch.tensor<[1],f32>
func @flatten_rank0(%arg0: !torch.tensor<[],f32>) -> !torch.tensor {
  %end = torch.constant.int -1
  %start = torch.constant.int 0
  %0 = torch.aten.flatten.using_ints %arg0, %start, %end : !torch.tensor<[],f32>, !torch.int, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func @torch.aten.unsqueeze$basic(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<[1],f32>
func @torch.aten.unsqueeze$basic(%arg0: !torch.tensor<[],f32>) -> !torch.tensor {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.tensor<[],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$basic_negative(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<[1],f32>
func @torch.aten.unsqueeze$basic_negative(%arg0: !torch.tensor<[],f32>) -> !torch.tensor {
  %int-1 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int-1 : !torch.tensor<[],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$invalid(
// CHECK:           torch.aten.unsqueeze {{.*}} !torch.tensor<*,f32>
func @torch.aten.unsqueeze$invalid(%arg0: !torch.tensor<[],f32>) -> !torch.tensor {
  %int1 = torch.constant.int 1
  %0 = torch.aten.unsqueeze %arg0, %int1 : !torch.tensor<[],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$invalid_negative(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<*,f32>
func @torch.aten.unsqueeze$invalid_negative(%arg0: !torch.tensor<[],f32>) -> !torch.tensor {
  %int-2 = torch.constant.int -2
  %0 = torch.aten.unsqueeze %arg0, %int-2 : !torch.tensor<[],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$higher_rank_front(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<[1,2,3,4],f32>
func @torch.aten.unsqueeze$higher_rank_front(%arg0: !torch.tensor<[2,3,4],f32>) -> !torch.tensor {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.tensor<[2,3,4],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$higher_rank_back(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<[2,3,4,1],f32>
func @torch.aten.unsqueeze$higher_rank_back(%arg0: !torch.tensor<[2,3,4],f32>) -> !torch.tensor {
  %int-1 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int-1 : !torch.tensor<[2,3,4],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$higher_rank_middle(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<[2,3,1,4],f32>
func @torch.aten.unsqueeze$higher_rank_middle(%arg0: !torch.tensor<[2,3,4],f32>) -> !torch.tensor {
  %int2 = torch.constant.int 2
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.tensor<[2,3,4],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// CHECK-LABEL:   func @torch.aten.unsqueeze$unknown_position(
// CHECK:           torch.aten.unsqueeze {{.*}} -> !torch.tensor<*,f32>
func @torch.aten.unsqueeze$unknown_position(%arg0: !torch.tensor<[2],f32>, %arg1: !torch.int) -> !torch.tensor {
  %0 = torch.aten.unsqueeze %arg0, %arg1 : !torch.tensor<[2],f32>, !torch.int -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// CHECK-LABEL: func @f
func @f(%arg0: !torch.vtensor<[4,6,3],f32>, %arg1: !torch.vtensor<[1,1,3],f32>, %arg2: !torch.vtensor<[?,3],f32>) {
  %int1 = torch.constant.int 1
  // CHECK: torch.aten.add{{.*}} -> !torch.vtensor<[?,?,?],f32>
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,6,3],f32>, !torch.vtensor<[1,1,3],f32>, !torch.int -> !torch.vtensor
  // CHECK: torch.aten.add{{.*}} -> !torch.vtensor<[?,?,?],f32>
  %1 = torch.aten.add.Tensor %arg0, %arg2, %int1 : !torch.vtensor<[4,6,3],f32>, !torch.vtensor<[?,3],f32>, !torch.int -> !torch.vtensor
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

// -----

// CHECK-LABEL:   builtin.func @f(
// CHECK-SAME:                    %[[TENSOR:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[TENSOR]] : !torch.tensor to !torch.optional<!torch.tensor>
// CHECK:           %[[RET:.*]] = torch.aten.__isnot__ %[[TENSOR]], %[[NONE]] : !torch.tensor, !torch.none -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool

func @f(%arg : !torch.tensor) -> !torch.bool {
  %none = torch.constant.none
  %optional = "torch.derefine"(%arg) : (!torch.tensor) -> !torch.optional<!torch.tensor>
  %ret = "torch.aten.__isnot__"(%optional, %none) : (!torch.optional<!torch.tensor>, !torch.none) -> !torch.bool
  return %ret: !torch.bool
}

// -----

// CHECK-LABEL:   builtin.func @aten.arange.start$int64_dtype(
// CHECK-SAME:                    %[[START:.*]]: !torch.int,
// CHECK-SAME:                    %[[END:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T:.*]] = torch.aten.arange.start
// CHECK-SAME:         %[[START]], %[[END]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:         !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none
// CHECK-SAME:         -> !torch.vtensor<[?],si64>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.vtensor<[?],si64> to !torch.vtensor
// CHECK:           return %[[RET]] : !torch.vtensor

func @aten.arange.start$int64_dtype(%start: !torch.int, %end: !torch.int) -> !torch.vtensor {
  %none = torch.constant.none
  %ret = torch.aten.arange.start %start, %end, %none, %none, %none, %none: !torch.int, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.arange.start$float32_dtype(
// CHECK-SAME:                    %[[START:.*]]: !torch.float,
// CHECK-SAME:                    %[[END:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T:.*]] = torch.aten.arange.start
// CHECK-SAME:         %[[START]], %[[END]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:         !torch.float, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none
// CHECK-SAME:         -> !torch.vtensor<[?],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.vtensor<[?],f32> to !torch.vtensor
// CHECK:           return %[[RET]] : !torch.vtensor

func @aten.arange.start$float32_dtype(%start: !torch.float, %end: !torch.int) -> !torch.vtensor {
  %none = torch.constant.none
  %ret = torch.aten.arange.start %start, %end, %none, %none, %none, %none: !torch.float, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.arange.start$specified_dtype(
// CHECK-SAME:                                                    %[[END:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[CST6:.*]] = torch.constant.int 6
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T:.*]] = torch.aten.arange
// CHECK-SAME:         %[[END]], %[[CST6]], %[[NONE]], %[[NONE]], %[[NONE]] :
// CHECK-SAME:         !torch.int, !torch.int, !torch.none, !torch.none, !torch.none
// CHECK-SAME:         -> !torch.vtensor<[?],f32>
// CHECK:           %[[RET:.*]] = torch.tensor_static_info_cast %[[T]] : !torch.vtensor<[?],f32> to !torch.vtensor
// CHECK:           return %[[RET]] : !torch.vtensor

func @aten.arange.start$specified_dtype(%end: !torch.int) -> !torch.vtensor {
  %int6 = torch.constant.int 6
  %none = torch.constant.none
  %ret = torch.aten.arange %end, %int6, %none, %none, %none: !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.sum.dim_IntList(
// CHECK-SAME:                                       %[[T:.*]]: !torch.vtensor<[2,3,?],si64>) -> !torch.vtensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[DIMLIST:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT_NEG1]]
// CHECK-SAME:        : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.sum.dim_IntList %[[T]], %[[DIMLIST]], %[[FALSE]], %[[NONE]]
// CHECK-SAME:        : !torch.vtensor<[2,3,?],si64>, !torch.list<!torch.int>, !torch.bool, !torch.none
// CHECK-SAME:        -> !torch.vtensor<[3],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<[3],si64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.sum.dim_IntList(%t: !torch.vtensor<[2,3,?],si64>) -> !torch.vtensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int-1 = torch.constant.int -1
  %dimList = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %ret = torch.aten.sum.dim_IntList %t, %dimList, %false, %none : !torch.vtensor<[2,3,?],si64>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.sum.dim_IntList$keepdim(
// CHECK-SAME:                                               %[[T:.*]]: !torch.vtensor<[2,3,?],si64>) -> !torch.vtensor {
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool true
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[DIMLIST:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT_NEG1]] :
// CHECK-SAME:        (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.sum.dim_IntList
// CHECK-SAME:        %[[T]], %[[DIMLIST]], %[[KEEPDIM]], %[[NONE]]
// CHECK-SAME:        : !torch.vtensor<[2,3,?],si64>, !torch.list<!torch.int>, !torch.bool, !torch.none
// CHECK-SAME:        -> !torch.vtensor<[1,3,1],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] :
// CHECK-SAME:        !torch.vtensor<[1,3,1],si64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.sum.dim_IntList$keepdim(%t: !torch.vtensor<[2,3,?],si64>) -> !torch.vtensor {
  %true = torch.constant.bool true
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int-1 = torch.constant.int -1
  %dimList = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %ret = torch.aten.sum.dim_IntList %t, %dimList, %true, %none : !torch.vtensor<[2,3,?],si64>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----
// CHECK-LABEL:   builtin.func @aten.sum.dim_IntList$unknown_position(
// CHECK-SAME:                                       %[[T:.*]]: !torch.vtensor<[2,3,?],si64>,
// CHECK-SAME:                                       %[[DIM:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[KEEPDIM:.*]] = torch.constant.bool false
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[DIMLIST:.*]] = torch.prim.ListConstruct %[[DIM]], %[[INT_NEG1]] : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.sum.dim_IntList %[[T]], %[[DIMLIST]], %[[KEEPDIM]], %[[NONE]] : !torch.vtensor<[2,3,?],si64>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.vtensor<[?],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<[?],si64> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.sum.dim_IntList$unknown_position(%t: !torch.vtensor<[2,3,?],si64>, %dim0: !torch.int) -> !torch.vtensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %int-1 = torch.constant.int -1
  %dimList = torch.prim.ListConstruct %dim0, %int-1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
  %ret = torch.aten.sum.dim_IntList %t, %dimList, %false, %none : !torch.vtensor<[2,3,?],si64>, !torch.list<!torch.int>, !torch.bool, !torch.none -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.any.dim(
// CHECK-SAME:                               %[[T:.*]]: !torch.vtensor<[2,3,?],i1>) -> !torch.vtensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[RET:.*]] = torch.aten.any.dim %[[T]], %[[INT_NEG1]], %[[FALSE]] : !torch.vtensor<[2,3,?],i1>, !torch.int, !torch.bool -> !torch.vtensor<[2,3],i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<[2,3],i1> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.any.dim(%t: !torch.vtensor<[2,3,?],i1>) -> !torch.vtensor {
  %false = torch.constant.bool false
  %int-1 = torch.constant.int -1
  %ret = torch.aten.any.dim %t, %int-1, %false : !torch.vtensor<[2,3,?],i1>, !torch.int, !torch.bool -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.any.dim$keepdim(
// CHECK-SAME:                               %[[T:.*]]: !torch.vtensor<[2,3,?],i1>) -> !torch.vtensor {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[RET:.*]] = torch.aten.any.dim %[[T]], %[[INT_NEG1]], %[[TRUE]] : !torch.vtensor<[2,3,?],i1>, !torch.int, !torch.bool -> !torch.vtensor<[2,3,1],i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<[2,3,1],i1> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.any.dim$keepdim(%t: !torch.vtensor<[2,3,?],i1>) -> !torch.vtensor {
  %true = torch.constant.bool true
  %int-1 = torch.constant.int -1
  %ret = torch.aten.any.dim %t, %int-1, %true : !torch.vtensor<[2,3,?],i1>, !torch.int, !torch.bool -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.any.dim$unknown_position(
// CHECK-SAME:                               %[[T:.*]]: !torch.vtensor<[2,3,?],i1>,
// CHECK-SAME:                               %[[DIM:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[RET:.*]] = torch.aten.any.dim %[[T]], %[[DIM]], %[[TRUE]] : !torch.vtensor<[2,3,?],i1>, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?],i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<[?,?,?],i1> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.any.dim$unknown_position(%t: !torch.vtensor<[2,3,?],i1>, %dim: !torch.int) -> !torch.vtensor {
  %true = torch.constant.bool true
  %ret = torch.aten.any.dim %t, %dim, %true : !torch.vtensor<[2,3,?],i1>, !torch.int, !torch.bool -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.any(
// CHECK-SAME:                           %[[T:.*]]: !torch.vtensor<[2,3,?],i1>) -> !torch.vtensor {
// CHECK:           %[[RET:.*]] = torch.aten.any %[[T]] : !torch.vtensor<[2,3,?],i1> -> !torch.vtensor<[1],i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.vtensor<[1],i1> to !torch.vtensor
// CHECK:           return %[[CAST]] : !torch.vtensor

func @aten.any(%t: !torch.vtensor<[2,3,?],i1>) -> !torch.vtensor {
  %ret = torch.aten.any %t: !torch.vtensor<[2,3,?],i1> -> !torch.vtensor
  return %ret : !torch.vtensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.transpose.int(
// CHECK-SAME:                                     %[[T:.*]]: !torch.tensor<[2,3,4,5],si64>) -> !torch.tensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[RET:.*]] = torch.aten.transpose.int %[[T]], %[[INT1]], %[[INT_NEG1]] : !torch.tensor<[2,3,4,5],si64>, !torch.int, !torch.int -> !torch.tensor<[2,5,4,3],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,5,4,3],si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @aten.transpose.int(%t: !torch.tensor<[2,3,4,5],si64>) -> !torch.tensor {
    %int1 = torch.constant.int 1
    %int-1 = torch.constant.int -1
    %ret = torch.aten.transpose.int %t, %int1, %int-1 : !torch.tensor<[2,3,4,5],si64>, !torch.int, !torch.int -> !torch.tensor
    return %ret: !torch.tensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.transpose.int$unknown_position(
// CHECK-SAME:                                     %[[T:.*]]: !torch.tensor<[2,3,4,5],si64>,
// CHECK-SAME:                                     %[[DIM0:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[RET:.*]] = torch.aten.transpose.int %[[T]], %[[DIM0]], %[[INT_NEG1]] : !torch.tensor<[2,3,4,5],si64>, !torch.int, !torch.int -> !torch.tensor<[?,?,?,?],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?,?,?,?],si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @aten.transpose.int$unknown_position(%t: !torch.tensor<[2,3,4,5],si64>, %dim0: !torch.int) -> !torch.tensor {
    %int-1 = torch.constant.int -1
    %ret = torch.aten.transpose.int %t, %dim0, %int-1 : !torch.tensor<[2,3,4,5],si64>, !torch.int, !torch.int -> !torch.tensor
    return %ret: !torch.tensor
}

// -----

// CHECK-LABEL:   builtin.func @aten.view(
// CHECK-SAME:                            %[[T:.*]]: !torch.tensor<[2,3,4,5],si64>) -> !torch.tensor {
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT_NEG1]]
// CHECK-SAME:        : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.view %[[T]], %[[SIZES]] :
// CHECK-SAME:        !torch.tensor<[2,3,4,5],si64>, !torch.list<!torch.int> -> !torch.tensor<[2,?],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,?],si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @aten.view(%t: !torch.tensor<[2,3,4,5],si64>) -> !torch.tensor {
    %int2 = torch.constant.int 2
    %int-1 = torch.constant.int -1
    %sizes = torch.prim.ListConstruct %int2, %int-1 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %ret = torch.aten.view %t, %sizes: !torch.tensor<[2,3,4,5],si64>, !torch.list<!torch.int> -> !torch.tensor
    return %ret: !torch.tensor
}

// -----

// CHECK-LABEL:   builtin.func @prim.if$refined_type_conflicting(
// CHECK-SAME:                                                   %[[NONE:.*]]: !torch.none) -> !torch.tensor {
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.tensor>
// CHECK:           %[[NOT_NONE:.*]] = torch.aten.__isnot__ %[[NONE]], %[[NONE]] : !torch.none, !torch.none -> !torch.bool
// CHECK:           %[[PRED:.*]] = torch.prim.If %[[NOT_NONE]] -> (!torch.tensor) {
// CHECK:             %[[T:.*]] = torch.prim.unchecked_cast %[[OPTIONAL]] : !torch.optional<!torch.tensor> -> !torch.tensor
// CHECK:             torch.prim.If.yield %[[T]] : !torch.tensor
// CHECK:           } else {
// CHECK:             %[[LITERAL:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<3x5xf32>) : !torch.tensor
// CHECK:             torch.prim.If.yield %[[LITERAL]] : !torch.tensor
// CHECK:           }
// CHECK:           return %[[PRED:.*]] : !torch.tensor

func @prim.if$refined_type_conflicting(%none: !torch.none) -> !torch.tensor {
  %optional = torch.derefine %none: !torch.none to !torch.optional<!torch.tensor>
  %pred = torch.aten.__isnot__ %optional, %none : !torch.optional<!torch.tensor>, !torch.none -> !torch.bool
  %res = torch.prim.If %pred -> (!torch.tensor) {
    %t = torch.prim.unchecked_cast %optional: !torch.optional<!torch.tensor> -> !torch.tensor
    torch.prim.If.yield %t: !torch.tensor
  } else {
    %t_cst = torch.tensor.literal(dense<0.0> : tensor<3x5xf32>) : !torch.tensor
    torch.prim.If.yield %t_cst: !torch.tensor
  }
  return %res: !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.tensor.float(
// CHECK-SAME:                                          %[[t:.*]]: !torch.float) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor.float %[[t]], %[[NONE]], %[[NONE]], %[[FALSE]] : !torch.float, !torch.none, !torch.none, !torch.bool -> !torch.tensor<[1],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[1],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.tensor.float(%t: !torch.float) -> !torch.tensor {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %ret = "torch.aten.tensor.float"(%t, %none, %none, %false) : (!torch.float, !torch.none, !torch.none, !torch.bool) -> !torch.tensor
  return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.tensor.float$specified_dtype(
// CHECK-SAME:                                          %[[t:.*]]: !torch.float) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[CST11:.*]] = torch.constant.int 11
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor.float %[[t]], %[[CST11]], %[[NONE]], %[[FALSE]] : !torch.float, !torch.int, !torch.none, !torch.bool -> !torch.tensor<[1],i1>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[1],i1> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.tensor.float$specified_dtype(%t: !torch.float) -> !torch.tensor {
  %none = torch.constant.none
  %int11 = torch.constant.int 11
  %false = torch.constant.bool false
  %ret = "torch.aten.tensor.float"(%t, %int11, %none, %false) : (!torch.float, !torch.int, !torch.none, !torch.bool) -> !torch.tensor
  return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.tensor(
// CHECK-SAME:        %[[DATA:.*]]: !torch.list<!torch.list<!torch.float>>) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor %[[DATA]], %[[NONE]], %[[NONE]], %[[FALSE]]
// CHECK-SAME:        : !torch.list<!torch.list<!torch.float>>, !torch.none, !torch.none, !torch.bool
// CHECK-SAME:        -> !torch.tensor<[?,?],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?,?],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.tensor(%t: !torch.list<!torch.list<!torch.float>>) -> !torch.tensor {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %ret = torch.aten.tensor %t, %none, %none, %false : !torch.list<!torch.list<!torch.float>>, !torch.none, !torch.none, !torch.bool -> !torch.tensor
    return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.tensor$empty_list() -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[DATA:.*]] = torch.prim.ListConstruct  : () -> !torch.list<!torch.float>
// CHECK:           %[[RET:.*]] = torch.aten.tensor %[[DATA]], %[[NONE]], %[[NONE]], %[[FALSE]] : !torch.list<!torch.float>, !torch.none, !torch.none, !torch.bool -> !torch.tensor<[?],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.tensor$empty_list() -> !torch.tensor {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %data = torch.prim.ListConstruct : () -> !torch.list<!torch.float>
    %ret = torch.aten.tensor %data, %none, %none, %false : !torch.list<!torch.float>, !torch.none, !torch.none, !torch.bool -> !torch.tensor
    return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.tensor$specified_dtype(
// CHECK-SAME:        %[[DATA:.*]]: !torch.list<!torch.list<!torch.float>>) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.tensor %[[DATA]], %[[INT4]], %[[NONE]], %[[FALSE]] : !torch.list<!torch.list<!torch.float>>, !torch.int, !torch.none, !torch.bool -> !torch.tensor<[?,?],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?,?],si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.tensor$specified_dtype(%t: !torch.list<!torch.list<!torch.float>>) -> !torch.tensor {
    %none = torch.constant.none
    %int4 = torch.constant.int 4
    %false = torch.constant.bool false
    %ret = torch.aten.tensor %t, %int4, %none, %false : !torch.list<!torch.list<!torch.float>>, !torch.int, !torch.none, !torch.bool -> !torch.tensor
    return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.zeros(
// CHECK-SAME:        %[[DIM0:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[ZEROS:.*]] = torch.aten.zeros %[[SIZES]], %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]] : !torch.list<!torch.int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor<[?,2],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[ZEROS]] : !torch.tensor<[?,2],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.zeros(%dim0: !torch.int) -> !torch.tensor {
    %none = torch.constant.none
    %int2 = torch.constant.int 2
    %sizesList = torch.prim.ListConstruct %dim0, %int2 : (!torch.int, !torch.int) -> !torch.list<!torch.int>
    %ret = torch.aten.zeros %sizesList, %none, %none, %none, %none : !torch.list<!torch.int>, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
    return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.index_select(
// CHECK-SAME:                                          %[[INPUT:.*]]: !torch.tensor<[2,3,4],f32>,
// CHECK-SAME:                                          %[[INDEXES:.*]]: !torch.tensor<[2],si64>) -> !torch.tensor {
// CHECK:           %[[DIM:.*]] = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.aten.index_select %[[INPUT]], %[[DIM]], %[[INDEXES]] : !torch.tensor<[2,3,4],f32>, !torch.int, !torch.tensor<[2],si64> -> !torch.tensor<[2,2,4],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,2,4],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.index_select(%input: !torch.tensor<[2,3,4], f32>, %index: !torch.tensor<[2], si64>) -> !torch.tensor {
      %dim = torch.constant.int 1
      %ret = torch.aten.index_select %input, %dim, %index : !torch.tensor<[2,3,4], f32>, !torch.int, !torch.tensor<[2], si64> -> !torch.tensor
      return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.index_select$unknown_indexes(
// CHECK-SAME:                                                      %[[INPUT:.*]]: !torch.tensor<[2,3,4],f32>,
// CHECK-SAME:                                                      %[[INDEXES:.*]]: !torch.tensor<[?],si64>) -> !torch.tensor {
// CHECK:           %[[DIM:.*]] = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.aten.index_select %[[INPUT]], %[[DIM]], %[[INDEXES]] : !torch.tensor<[2,3,4],f32>, !torch.int, !torch.tensor<[?],si64> -> !torch.tensor<[2,?,4],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,?,4],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.index_select$unknown_indexes(%input: !torch.tensor<[2,3,4], f32>, %index: !torch.tensor<[?], si64>) -> !torch.tensor {
      %dim = torch.constant.int 1
      %ret = torch.aten.index_select %input, %dim, %index : !torch.tensor<[2,3,4], f32>, !torch.int, !torch.tensor<[?], si64> -> !torch.tensor
      return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.index_select$unknown_dim(
// CHECK-SAME:                                                          %[[INPUT:.*]]: !torch.tensor<[2,3,4],f32>,
// CHECK-SAME:                                                          %[[DIM:.*]]: !torch.int,
// CHECK-SAME:                                                          %[[INDEXES:.*]]: !torch.tensor<[?],si64>) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten.index_select %[[INPUT]], %[[DIM]], %[[INDEXES]] : !torch.tensor<[2,3,4],f32>, !torch.int, !torch.tensor<[?],si64> -> !torch.tensor<[?,?,?],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?,?,?],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.index_select$unknown_dim(%input: !torch.tensor<[2,3,4], f32>, %dim: !torch.int, %index: !torch.tensor<[?], si64>) -> !torch.tensor {
      %ret = torch.aten.index_select %input, %dim, %index : !torch.tensor<[2,3,4], f32>, !torch.int, !torch.tensor<[?], si64> -> !torch.tensor
      return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.select.int(
// CHECK-SAME:                                        %[[INPUT:.*]]: !torch.tensor<[2,3,4],f32>,
// CHECK-SAME:                                        %[[INDEX:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[DIM:.*]] = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.aten.select.int %[[INPUT]], %[[DIM]], %[[INDEX]] : !torch.tensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.tensor<[2,1,4],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,1,4],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.select.int(%input: !torch.tensor<[2,3,4], f32>, %index: !torch.int) -> !torch.tensor {
      %dim = torch.constant.int 1
      %ret = torch.aten.select.int %input, %dim, %index : !torch.tensor<[2,3,4], f32>, !torch.int, !torch.int -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.type_as(
// CHECK-SAME:                                     %[[INPUT:.*]]: !torch.tensor<[?],si64>,
// CHECK-SAME:                                     %[[OTHER:.*]]: !torch.tensor<[?,2],f32>) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten.type_as %[[INPUT]], %[[OTHER]] : !torch.tensor<[?],si64>, !torch.tensor<[?,2],f32> -> !torch.tensor<[?],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.type_as(%self: !torch.tensor<[?], si64>, %other: !torch.tensor<[?,2],f32>) -> !torch.tensor {
      %ret = torch.aten.type_as %self, %other : !torch.tensor<[?], si64>, !torch.tensor<[?,2],f32> -> !torch.tensor
      return %ret: !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.gather(
// CHECK-SAME:                                        %[[INPUT:.*]]: !torch.tensor<[2,3,4],f32>,
// CHECK-SAME:                                        %[[DIM:.*]]: !torch.int,
// CHECK-SAME:                                        %[[INDEXES:.*]]: !torch.tensor<[1,2,3],si64>) -> !torch.tensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[RET:.*]] = torch.aten.gather %[[INPUT]], %[[DIM]], %[[INDEXES]], %[[FALSE]] : !torch.tensor<[2,3,4],f32>, !torch.int, !torch.tensor<[1,2,3],si64>, !torch.bool -> !torch.tensor<[1,2,3],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[1,2,3],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.gather(%input: !torch.tensor<[2,3,4], f32>, %dim: !torch.int, %index: !torch.tensor<[1,2,3], si64>) -> !torch.tensor {
      %false = torch.constant.bool false
      %ret = torch.aten.gather %input, %dim, %index, %false : !torch.tensor<[2,3,4], f32>, !torch.int, !torch.tensor<[1,2,3], si64>, !torch.bool -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.expand(
// CHECK-SAME:                                    %[[INPUT:.*]]: !torch.tensor<[2,1,4],f32>) -> !torch.tensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[INT5:.*]] = torch.constant.int 5
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct %[[INT_NEG1]], %[[INT5]], %[[INT4]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.expand %[[INPUT]], %[[SIZES]], %[[FALSE]] : !torch.tensor<[2,1,4],f32>, !torch.list<!torch.int>, !torch.bool -> !torch.tensor<[2,5,4],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,5,4],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.expand(%input: !torch.tensor<[2,1,4], f32>) -> !torch.tensor {
      %false = torch.constant.bool false
      %int-1 = torch.constant.int -1
      %int5 = torch.constant.int 5
      %int4 = torch.constant.int 4
      %size = torch.prim.ListConstruct %int-1, %int5, %int4: (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
      %ret = torch.aten.expand %input, %size, %false : !torch.tensor<[2,1,4], f32>, !torch.list<!torch.int>, !torch.bool -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.expand$unknown_sizes(
// CHECK-SAME:                                                  %[[INPUT:.*]]: !torch.tensor<[2,1,4],f32>,
// CHECK-SAME:                                                  %[[SIZEX:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[INT_NEG1:.*]] = torch.constant.int -1
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[SIZES:.*]] = torch.prim.ListConstruct %[[INT_NEG1]], %[[SIZEX]], %[[INT4]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.expand %[[INPUT]], %[[SIZES]], %[[FALSE]] : !torch.tensor<[2,1,4],f32>, !torch.list<!torch.int>, !torch.bool -> !torch.tensor<[2,?,4],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,?,4],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
// CHECK:         }
func @torch.aten.expand$unknown_sizes(%input: !torch.tensor<[2,1,4], f32>, %index: !torch.int) -> !torch.tensor {
      %false = torch.constant.bool false
      %int-1 = torch.constant.int -1
      %int4 = torch.constant.int 4
      %size = torch.prim.ListConstruct %int-1, %index, %int4: (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
      %ret = torch.aten.expand %input, %size, %false : !torch.tensor<[2,1,4], f32>, !torch.list<!torch.int>, !torch.bool -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.repeat(
// CHECK-SAME:                                    %[[INPUT:.*]]: !torch.tensor<[2,1,4],f32>,
// CHECK-SAME:                                    %[[REPEATX:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[INT4:.*]] = torch.constant.int 4
// CHECK:           %[[REPEATS:.*]] = torch.prim.ListConstruct %[[INT1]], %[[REPEATX]], %[[INT4]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
// CHECK:           %[[RET:.*]] = torch.aten.repeat %[[INPUT]], %[[REPEATS]] : !torch.tensor<[2,1,4],f32>, !torch.list<!torch.int> -> !torch.tensor<[2,?,16],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,?,16],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.repeat(%input: !torch.tensor<[2,1,4], f32>, %repeat: !torch.int) -> !torch.tensor {
      %int1 = torch.constant.int 1
      %int4 = torch.constant.int 4
      %repeats = torch.prim.ListConstruct %int1, %repeat, %int4: (!torch.int, !torch.int, !torch.int) -> !torch.list<!torch.int>
      %ret = torch.aten.repeat %input, %repeats: !torch.tensor<[2,1,4], f32>, !torch.list<!torch.int> -> !torch.tensor
      return %ret : !torch.tensor
}

// ----

// CHECK-LABEL:   builtin.func @torch.aten.cat(
// CHECK-SAME:                                 %[[T1:.*]]: !torch.tensor<[?,1,4],f32>,
// CHECK-SAME:                                 %[[T2:.*]]: !torch.tensor<[2,3,4],f32>) -> !torch.tensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[TENSORS:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]] : (!torch.tensor<[?,1,4],f32>, !torch.tensor<[2,3,4],f32>) -> !torch.list<!torch.tensor>
// CHECK:           %[[RET:.*]] = torch.aten.cat %[[TENSORS]], %[[INT1]] : !torch.list<!torch.tensor>, !torch.int -> !torch.tensor<[2,?,4],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,?,4],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.cat(%t0: !torch.tensor<[?,1,4], f32>, %t1: !torch.tensor<[2,3,4], f32>) -> !torch.tensor {
      %int1 = torch.constant.int 1
      %tensorList = torch.prim.ListConstruct %t0, %t1: (!torch.tensor<[?,1,4], f32>, !torch.tensor<[2,3,4], f32>) -> !torch.list<!torch.tensor>
      %ret = torch.aten.cat %tensorList, %int1 : !torch.list<!torch.tensor>, !torch.int -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.cat$unknown_dim(
// CHECK-SAME:                                 %[[T1:.*]]: !torch.tensor<[?,1,4],f32>,
// CHECK-SAME:                                 %[[T2:.*]]: !torch.tensor<[2,3,4],f32>,
// CHECK-SAME:                                 %[[DIM:.*]]: !torch.int) -> !torch.tensor {
// CHECK:           %[[TENSORS:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]] : (!torch.tensor<[?,1,4],f32>, !torch.tensor<[2,3,4],f32>) -> !torch.list<!torch.tensor>
// CHECK:           %[[RET:.*]] = torch.aten.cat %[[TENSORS]], %[[DIM]] : !torch.list<!torch.tensor>, !torch.int -> !torch.tensor<[?,?,?],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?,?,?],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor

func @torch.aten.cat$unknown_dim(%t0: !torch.tensor<[?,1,4], f32>, %t1: !torch.tensor<[2,3,4], f32>, %dim: !torch.int) -> !torch.tensor {
      %tensorList = torch.prim.ListConstruct %t0, %t1: (!torch.tensor<[?,1,4], f32>, !torch.tensor<[2,3,4], f32>) -> !torch.list<!torch.tensor>
      %ret = torch.aten.cat %tensorList, %dim: !torch.list<!torch.tensor>, !torch.int -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten._shape_as_tensor(
// CHECK-SAME:                                 %[[INPUT:.*]]: !torch.tensor<[?,1,4],f32>) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten._shape_as_tensor %[[INPUT]] : !torch.tensor<[?,1,4],f32> -> !torch.tensor<[3],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[3],si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
// CHECK:         }
func @torch.aten._shape_as_tensor(%input: !torch.tensor<[?,1,4], f32>) -> !torch.tensor {
      %ret= torch.aten._shape_as_tensor %input : !torch.tensor<[?,1,4], f32> -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten._shape_as_tensor$unknown_input_shape(
// CHECK-SAME:                                 %[[INPUT:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:           %[[RET:.*]] = torch.aten._shape_as_tensor %[[INPUT]] : !torch.tensor -> !torch.tensor<[?],si64>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[?],si64> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
// CHECK:         }
func @torch.aten._shape_as_tensor$unknown_input_shape(%input: !torch.tensor) -> !torch.tensor {
      %ret= torch.aten._shape_as_tensor %input : !torch.tensor -> !torch.tensor
      return %ret : !torch.tensor
}

// ----
// CHECK-LABEL:   builtin.func @torch.aten.embedding(
// CHECK-SAME:                                       %[[INPUT:.*]]: !torch.tensor<[104,512],f32>,
// CHECK-SAME:                                       %[[INDEXES:.*]]: !torch.tensor<[2,3],si64>) -> !torch.tensor {
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[PADDING_IDX:.*]] = torch.constant.int 1
// CHECK:           %[[RET:.*]] = torch.aten.embedding %[[INPUT]], %[[INDEXES]], %[[PADDING_IDX]], %[[FALSE]], %[[FALSE]] : !torch.tensor<[104,512],f32>, !torch.tensor<[2,3],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.tensor<[2,3,512],f32>
// CHECK:           %[[CAST:.*]] = torch.tensor_static_info_cast %[[RET]] : !torch.tensor<[2,3,512],f32> to !torch.tensor
// CHECK:           return %[[CAST]] : !torch.tensor
func @torch.aten.embedding(%weight: !torch.tensor<[104,512],f32>, %indices: !torch.tensor<[2,3], si64>) -> !torch.tensor {
       %false = torch.constant.bool false
       %int1 = torch.constant.int 1
       %ret = torch.aten.embedding %weight, %indices, %int1, %false, %false : !torch.tensor<[104,512],f32>, !torch.tensor<[2,3], si64>, !torch.int, !torch.bool, !torch.bool -> !torch.tensor
       return %ret: !torch.tensor
}
