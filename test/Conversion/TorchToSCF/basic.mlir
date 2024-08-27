// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-to-scf| FileCheck %s

// CHECK-LABEL:   func.func @torch.prim.if(
// CHECK-SAME:                        %[[VAL_0:.*]]: !torch.bool) -> !torch.int {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_i1 %[[VAL_0]]
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = scf.if %[[VAL_1]] -> (i64) {
// CHECK:             scf.yield %[[VAL_3]] : i64
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_5]] : i64
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = torch_c.from_i64 %[[VAL_8:.*]]
// CHECK:           return %[[VAL_7]] : !torch.int
func.func @torch.prim.if(%arg0: !torch.bool) -> !torch.int {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %0 = torch.prim.If %arg0 -> (!torch.int) {
    torch.prim.If.yield %int2 : !torch.int
  } else {
    torch.prim.If.yield %int1 : !torch.int
  }
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @aten.prim.if$nested(
// CHECK-SAME:                              %[[VAL_0:.*]]: !torch.bool,
// CHECK-SAME:                              %[[VAL_1:.*]]: !torch.bool) -> !torch.int {
// CHECK-DAG:       %[[VAL_2:.*]] = torch_c.to_i1 %[[VAL_0]]
// CHECK-DAG:       %[[VAL_3:.*]] = torch_c.to_i1 %[[VAL_1]]
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_7:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_8:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_9:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_10:.*]] = scf.if %[[VAL_2]] -> (i64) {
// CHECK:             %[[VAL_11:.*]] = scf.if %[[VAL_3]] -> (i64) {
// CHECK:               scf.yield %[[VAL_5]] : i64
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_7]] : i64
// CHECK:             }
// CHECK:             scf.yield %[[VAL_12:.*]] : i64
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_9]] : i64
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = torch_c.from_i64 %[[VAL_14:.*]]
// CHECK:           return %[[VAL_13]] : !torch.int
func.func @aten.prim.if$nested(%arg0: !torch.bool, %arg1: !torch.bool) -> !torch.int {
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.prim.If %arg0 -> (!torch.int) {
    %1 = torch.prim.If %arg1 -> (!torch.int) {
      torch.prim.If.yield %int2 : !torch.int
    } else {
      torch.prim.If.yield %int3 : !torch.int
    }
    torch.prim.If.yield %1 : !torch.int
  } else {
    torch.prim.If.yield %int4 : !torch.int
  }
  return %0 : !torch.int
}

// CHECK-LABEL: func.func @torch.prim.loop$while
// CHECK-SAME:  (%[[ARG0:.*]]: !torch.int) -> !torch.float {
// CHECK:         %[[TORCH_FLOAT_VAL:.*]] = torch.constant.float
// CHECK-NEXT:    %[[FLOAT_VAL:.*]] = torch_c.to_f64 %[[TORCH_FLOAT_VAL]]
// CHECK-NEXT:    %[[MAX_TRIP_COUNT:.*]] = torch.constant.int 9223372036854775807
// CHECK-NEXT:    %[[TORCH_CONDITION:.*]] = torch.aten.lt.float_int %[[TORCH_FLOAT_VAL]], %[[ARG0]]
// CHECK-NEXT:    %[[CONDITION:.*]] = torch_c.to_i1 %[[TORCH_CONDITION]]
// CHECK-NEXT:    %[[LOOP:.*]] = scf.while
// CHECK-SAME:    (%[[LOOP_CONDITION:.*]] = %[[CONDITION]], %[[LOOP_ARG:.*]] = %[[FLOAT_VAL]]) : (i1, f64) -> f64 {
// CHECK-NEXT:      scf.condition(%[[LOOP_CONDITION]]) %[[LOOP_ARG]]
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[BLOCK_ARG:.*]]: f64):
// CHECK-NEXT:      %[[TORCH_BLOCK_ARG:.*]] = torch_c.from_f64 %[[BLOCK_ARG]]
// CHECK-NEXT:      %[[TORCH_VAL:.*]] = torch.aten.mul.float %[[TORCH_BLOCK_ARG]], %[[TORCH_BLOCK_ARG]]
// CHECK-NEXT:      %[[TORCH_BLOCK_CONDITION:.*]] = torch.aten.lt.float_int %[[TORCH_VAL]], %[[ARG0]]
// CHECK-NEXT:      %[[BLOCK_CONDITION:.*]] = torch_c.to_i1 %[[TORCH_BLOCK_CONDITION]]
// CHECK-NEXT:      %[[VAL:.*]] = torch_c.to_f64 %[[TORCH_VAL]]
// CHECK-NEXT:      scf.yield %[[BLOCK_CONDITION]], %[[VAL]] : i1, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[TORCH_LOOP:.*]] = torch_c.from_f64 %[[LOOP]]
// CHECK-NEXT:    return %[[TORCH_LOOP]] : !torch.float
func.func @torch.prim.loop$while(%arg0: !torch.int) -> !torch.float {
  %float3.200000e00 = torch.constant.float 3.200000e+00
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %0 = torch.aten.lt.float_int %float3.200000e00, %arg0 : !torch.float, !torch.int -> !torch.bool
  %1 = torch.prim.Loop %int9223372036854775807, %0, init(%float3.200000e00) {
  ^bb0(%arg1: !torch.int, %arg2: !torch.float):
    %2 = torch.aten.mul.float %arg2, %arg2 : !torch.float, !torch.float -> !torch.float
    %3 = torch.aten.lt.float_int %2, %arg0 : !torch.float, !torch.int -> !torch.bool
    torch.prim.Loop.condition %3, iter(%2 : !torch.float)
  } : (!torch.int, !torch.bool, !torch.float) -> !torch.float
  return %1 : !torch.float
}

// CHECK-LABEL: func.func @torch.prim.loop$while_with_multiple_values
// CHECK-SAME:  () -> (!torch.float, !torch.float) {
// CHECK:         %[[TORCH_FLOAT_VAL_0:.*]] = torch.constant.float
// CHECK-NEXT:    %[[FLOAT_VAL_0:.*]] = torch_c.to_f64 %[[TORCH_FLOAT_VAL_0]]
// CHECK-NEXT:    %[[MAX_TRIP_COUNT:.*]] = torch.constant.int 9223372036854775807
// CHECK-NEXT:    %[[TORCH_FLOAT_VAL_1:.*]] = torch.constant.float
// CHECK-NEXT:    %[[FLOAT_VAL_1:.*]] = torch_c.to_f64 %[[TORCH_FLOAT_VAL_1]]
// CHECK-NEXT:    %[[TORCH_CONDITION:.*]] = torch.aten.lt.float %[[TORCH_FLOAT_VAL_0]], %[[TORCH_FLOAT_VAL_1]]
// CHECK-NEXT:    %[[CONDITION:.*]] = torch_c.to_i1 %[[TORCH_CONDITION]]
// CHECK-NEXT:    %[[LOOP:.*]]:2 = scf.while
// CHECK-SAME:    (%[[LOOP_CONDITION:.*]] = %[[CONDITION]], %[[LOOP_ARG_0:.*]] = %[[FLOAT_VAL_0]], %[[LOOP_ARG_1:.*]] = %[[FLOAT_VAL_1]]) : (i1, f64, f64) -> (f64, f64) {
// CHECK-NEXT:      scf.condition(%[[LOOP_CONDITION]]) %[[LOOP_ARG_0]], %[[LOOP_ARG_1]]
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[BLOCK_ARG_0:.*]]: f64, %[[BLOCK_ARG_1:.*]]: f64):
// CHECK-NEXT:      %[[TORCH_BLOCK_ARG_0:.*]] = torch_c.from_f64 %[[BLOCK_ARG_0]]
// CHECK-NEXT:      %[[TORCH_BLOCK_ARG_1:.*]] = torch_c.from_f64 %[[BLOCK_ARG_1]]
// CHECK-NEXT:      %[[TORCH_VAL_0:.*]] = torch.aten.mul.float %[[TORCH_BLOCK_ARG_0]], %[[TORCH_BLOCK_ARG_0]]
// CHECK-NEXT:      %[[TORCH_BLOCK_CONDITION:.*]] = torch.aten.lt.float %[[TORCH_VAL_0]], %[[TORCH_BLOCK_ARG_1]]
// CHECK-NEXT:      %[[CONSTANT:.*]] = torch.constant.int -2
// CHECK-NEXT:      %[[TORCH_VAL_1:.*]] = torch.aten.add.float_int %[[TORCH_BLOCK_ARG_1]], %[[CONSTANT]]
// CHECK-NEXT:      %[[BLOCK_CONDITION:.*]] = torch_c.to_i1 %[[TORCH_BLOCK_CONDITION]]
// CHECK-NEXT:      %[[VAL_0:.*]] = torch_c.to_f64 %[[TORCH_VAL_0]]
// CHECK-NEXT:      %[[VAL_1:.*]] = torch_c.to_f64 %[[TORCH_VAL_1]]
// CHECK-NEXT:      scf.yield %[[BLOCK_CONDITION]], %[[VAL_0]], %[[VAL_1]] : i1, f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[TORCH_LOOP_1:.*]] = torch_c.from_f64 %[[LOOP]]#1
// CHECK-NEXT:    %[[TORCH_LOOP_0:.*]] = torch_c.from_f64 %[[LOOP]]#0
// CHECK-NEXT:    return %[[TORCH_LOOP_0]], %[[TORCH_LOOP_1]] : !torch.float, !torch.float
func.func @torch.prim.loop$while_with_multiple_values() -> (!torch.float, !torch.float) {
  %float3.200000e00 = torch.constant.float 3.200000e+00
  %int9223372036854775807 = torch.constant.int 9223372036854775807
  %float9.0 = torch.constant.float 9.0
  %0 = torch.aten.lt.float %float3.200000e00, %float9.0 : !torch.float, !torch.float -> !torch.bool
  %1:2 = torch.prim.Loop %int9223372036854775807, %0, init(%float3.200000e00, %float9.0) {
  ^bb0(%arg1: !torch.int, %arg2: !torch.float, %arg3: !torch.float):
    %2 = torch.aten.mul.float %arg2, %arg2 : !torch.float, !torch.float -> !torch.float
    %3 = torch.aten.lt.float %2, %arg3 : !torch.float, !torch.float -> !torch.bool
    %4 = torch.constant.int -2
    %5 = torch.aten.add.float_int %arg3, %4 : !torch.float, !torch.int -> !torch.float
    torch.prim.Loop.condition %3, iter(%2, %5 : !torch.float, !torch.float)
  } : (!torch.int, !torch.bool, !torch.float, !torch.float) -> (!torch.float, !torch.float)
  return %1#0, %1#1 : !torch.float, !torch.float
}

// CHECK-LABEL: func.func @torch.prim.Loop$for
// CHECK-SAME:  (%[[TORCH_ARG0:.*]]: !torch.int) -> !torch.float {
// CHECK:         %[[ARG0:.*]] = torch_c.to_i64 %[[TORCH_ARG0]]
// CHECK-NEXT:    %{{.*}} = torch.constant.bool true
// CHECK-NEXT:    %[[TORCH_FLOAT:.*]] = torch.constant.float 0.000000e+00
// CHECK-NEXT:    %[[FLOAT:.*]] = torch_c.to_f64 %[[TORCH_FLOAT]]
// CHECK-NEXT:    %[[LOWER_BOUND:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[STEP:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[UPPER_BOUND:.*]] = arith.index_cast %[[ARG0]] : i64 to index
// CHECK-NEXT:    %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[LOWER_BOUND]] to %[[UPPER_BOUND]] step %[[STEP]]
// CHECK-SAME:      iter_args(%[[ITER_ARG:.*]] = %[[FLOAT]]) -> (f64) {
// CHECK-NEXT:      %[[IV_I64:.*]] = arith.index_cast %[[IV]] : index to i64
// CHECK-NEXT:      %[[TORCH_IV:.*]] = torch_c.from_i64 %[[IV_I64]]
// CHECK-NEXT:      %[[TORCH_ITER_ARG:.*]] = torch_c.from_f64 %[[ITER_ARG]]
// CHECK-NEXT:      %[[TORCH_VAL:.*]] = torch.aten.add.float_int %[[TORCH_ITER_ARG]], %[[TORCH_IV]]
// CHECK-NEXT:      %[[VAL:.*]] = torch_c.to_f64 %[[TORCH_VAL]]
// CHECK-NEXT:      scf.yield %[[VAL]] : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RETURN:.*]] = torch_c.from_f64 %[[LOOP]]
// CHECK-NEXT:    return %[[RETURN]] : !torch.float
// CHECK-NEXT:  }
func.func @torch.prim.Loop$for(%arg0: !torch.int) -> !torch.float {
  %true = torch.constant.bool true
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %0 = torch.prim.Loop %arg0, %true, init(%float0.000000e00) {
  ^bb0(%arg1: !torch.int, %arg2: !torch.float):
    %1 = torch.aten.add.float_int %arg2, %arg1 : !torch.float, !torch.int -> !torch.float
    torch.prim.Loop.condition %true, iter(%1 : !torch.float)
  } : (!torch.int, !torch.bool, !torch.float) -> !torch.float
  return %0 : !torch.float
}

// CHECK-LABEL: func.func @torch.prim.Loop$for_with_multiple_results
// CHECK-SAME:  (%[[TORCH_ARG0:.*]]: !torch.int) -> (!torch.float, !torch.float) {
// CHECK:         %[[ARG0:.*]] = torch_c.to_i64 %[[TORCH_ARG0]]
// CHECK-NEXT:    %{{.*}} = torch.constant.bool true
// CHECK-NEXT:    %[[TORCH_FLOAT_0:.*]] = torch.constant.float 0.000000e+00
// CHECK-NEXT:    %[[FLOAT_0:.*]] = torch_c.to_f64 %[[TORCH_FLOAT_0]]
// CHECK-NEXT:    %[[TORCH_FLOAT_1:.*]] = torch.constant.float 9.000000e+00
// CHECK-NEXT:    %[[FLOAT_1:.*]] = torch_c.to_f64 %[[TORCH_FLOAT_1]]
// CHECK-NEXT:    %[[LOWER_BOUND:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[STEP:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[UPPER_BOUND:.*]] = arith.index_cast %[[ARG0]] : i64 to index
// CHECK-NEXT:    %[[LOOP:.*]]:2 = scf.for %[[IV:.*]] = %[[LOWER_BOUND]] to %[[UPPER_BOUND]] step %[[STEP]]
// CHECK-SAME:      iter_args(%[[ITER_ARG_0:.*]] = %[[FLOAT_0]], %[[ITER_ARG_1:.*]] = %[[FLOAT_1]]) -> (f64, f64) {
// CHECK-NEXT:      %[[IV_I64:.*]] = arith.index_cast %[[IV]] : index to i64
// CHECK-NEXT:      %[[TORCH_IV:.*]] = torch_c.from_i64 %[[IV_I64]]
// CHECK-NEXT:      %[[TORCH_ITER_ARG_0:.*]] = torch_c.from_f64 %[[ITER_ARG_0]]
// CHECK-NEXT:      %[[TORCH_ITER_ARG_1:.*]] = torch_c.from_f64 %[[ITER_ARG_1]]
// CHECK-NEXT:      %[[TORCH_VAL_0:.*]] = torch.aten.add.float_int %[[TORCH_ITER_ARG_0]], %[[TORCH_IV]]
// CHECK-NEXT:      %[[TORCH_VAL_1:.*]] = torch.aten.mul.float %[[TORCH_ITER_ARG_1]], %[[TORCH_VAL_0]]
// CHECK-NEXT:      %[[VAL_0:.*]] = torch_c.to_f64 %[[TORCH_VAL_0]]
// CHECK-NEXT:      %[[VAL_1:.*]] = torch_c.to_f64 %[[TORCH_VAL_1]]
// CHECK-NEXT:      scf.yield %[[VAL_0]], %[[VAL_1]] : f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RETURN_1:.*]] = torch_c.from_f64 %[[LOOP]]#1
// CHECK-NEXT:    %[[RETURN_0:.*]] = torch_c.from_f64 %[[LOOP]]#0
// CHECK-NEXT:    return %[[RETURN_0]], %[[RETURN_1]] : !torch.float, !torch.float
// CHECK-NEXT:  }
func.func @torch.prim.Loop$for_with_multiple_results(%arg0: !torch.int) -> (!torch.float, !torch.float) {
  %true = torch.constant.bool true
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %float9.0 = torch.constant.float 9.0
  %0:2 = torch.prim.Loop %arg0, %true, init(%float0.000000e00, %float9.0) {
  ^bb0(%arg1: !torch.int, %arg2: !torch.float, %arg3: !torch.float):
    %1 = torch.aten.add.float_int %arg2, %arg1 : !torch.float, !torch.int -> !torch.float
    %2 = torch.aten.mul.float %arg3, %1 : !torch.float, !torch.float -> !torch.float
    torch.prim.Loop.condition %true, iter(%1, %2 : !torch.float, !torch.float)
  } : (!torch.int, !torch.bool, !torch.float, !torch.float) -> (!torch.float, !torch.float)
  return %0#0, %0#1 : !torch.float, !torch.float
}


// -----

// CHECK-LABEL:   func.func @torch.prim.Loop$for_with_tensor_arg() -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[LOOP_RESULT:.*]] = scf.for %[[LOOP_VARIABLE:.*]] = %[[RANGE_START:.*]] to %[[RANGE_END:.*]] step %[[RANGE_STEP:.*]] iter_args(%[[LOOP_TENSOR_ARG:.*]] = %[[LOOP_TENSOR_ARG_INIT_VAL:.*]]) -> (tensor<2x3xf32>) {
// CHECK:             %[[LOOP_TENSOR_ARG_TORCH_TENSOR:.*]] = torch_c.from_builtin_tensor %[[LOOP_TENSOR_ARG]]
// CHECK:           }
// CHECK:         }
func.func @torch.prim.Loop$for_with_tensor_arg() -> (!torch.vtensor<[2,3],f32>) {
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %int5 = torch.constant.int 5
    %int6 = torch.constant.int 6
    %none = torch.constant.none
    %0 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.zeros %0, %int6, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
    %2 = torch.aten.ones %0, %int6, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,3],f32>
    %3:1 = torch.prim.Loop %int5, %true, init(%1) {
    ^bb0(%arg1: !torch.int, %arg2: !torch.vtensor<[2,3],f32>):
        %4 = torch.aten.add.Tensor %arg2, %2, %int1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[2,3],f32>
        torch.prim.Loop.condition %true, iter(%4 : !torch.vtensor<[2,3],f32>)
    } : (!torch.int, !torch.bool, !torch.vtensor<[2,3],f32>) -> (!torch.vtensor<[2,3],f32>)
    return %3#0 : !torch.vtensor<[2,3],f32>
}
