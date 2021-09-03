// RUN: npcomp-opt -torch-refine-types -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL:   func @prim.if$branch_merge_type_tensor(
// CHECK-SAME:                                                   %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:                                                   %[[T1:.*]]: !torch.tensor,
// CHECK-SAME:                                                   %[[T2:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[MERGED:.*]] = torch.prim.If %[[PRED]] -> (!torch.optional<!torch.tensor>) {
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[T1]] : !torch.tensor to !torch.optional<!torch.tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<!torch.tensor>
// CHECK:           } else {
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[T2]] : !torch.tensor to !torch.optional<!torch.tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<!torch.tensor>
// CHECK:           }
// CHECK:           %[[REFINED:.*]] = torch.prim.unchecked_cast %[[MERGED:.*]] : !torch.optional<!torch.tensor> -> !torch.tensor
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[RET:.*]] = torch.aten.__isnot__ %[[REFINED]], %[[NONE]] : !torch.tensor, !torch.none -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool

func @prim.if$branch_merge_type_tensor(%pred: !torch.bool, %t0: !torch.tensor, %t1: !torch.tensor) -> !torch.bool {
  %res = torch.prim.If %pred -> (!torch.optional<!torch.tensor>) {
    %optional0 = torch.derefine %t0: !torch.tensor to !torch.optional<!torch.tensor>
    torch.prim.If.yield %optional0: !torch.optional<!torch.tensor>
  } else {
    %optional1 = torch.derefine %t1: !torch.tensor to !torch.optional<!torch.tensor>
    torch.prim.If.yield %optional1: !torch.optional<!torch.tensor>
  }
  %none = torch.constant.none
  %cmp = torch.aten.__isnot__ %res, %none : !torch.optional<!torch.tensor>, !torch.none -> !torch.bool
  return %cmp : !torch.bool
}

// -----

// CHECK-LABEL:   func @prim.if$branch_merge_type_optional(
// CHECK-SAME:                                                     %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:                                                     %[[T:.*]]: !torch.tensor) -> !torch.optional<!torch.tensor> {
// CHECK:           %[[MERGED:.*]] = torch.prim.If %[[PRED]] -> (!torch.optional<!torch.tensor>) {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<!torch.tensor>
// CHECK:           } else {
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[T]] : !torch.tensor to !torch.optional<!torch.tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<!torch.tensor>
// CHECK:           }
// CHECK:           return %[[MERGED:.*]] : !torch.optional<!torch.tensor>

func @prim.if$branch_merge_type_optional(%pred: !torch.bool, %t1: !torch.tensor) -> !torch.optional<!torch.tensor> {
  %res = torch.prim.If %pred -> (!torch.optional<!torch.tensor>) {
    %none = torch.constant.none
    %optional0 = torch.derefine %none: !torch.none to !torch.optional<!torch.tensor>
    torch.prim.If.yield %optional0: !torch.optional<!torch.tensor>
  } else {
    %optional1 = torch.derefine %t1: !torch.tensor to !torch.optional<!torch.tensor>
    torch.prim.If.yield %optional1: !torch.optional<!torch.tensor>
  }
  return %res: !torch.optional<!torch.tensor>
}

// -----

// CHECK-LABEL:   func @prim.loop$region_arg_to_internal(
// CHECK-SAME:                            %[[ARG_NONE:.*]]: !torch.none) -> !torch.optional<!torch.tensor> {
// CHECK:           %[[INT10:.*]] = torch.constant.int 10
// CHECK:           %[[INDV:.*]] = torch.constant.int 0
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[ARG_NONE]] : !torch.none to !torch.optional<!torch.tensor>
// CHECK:           %[[LOOP_RET:.*]] = torch.prim.Loop %[[INT10]], %[[TRUE]], init(%[[OPTIONAL]])  {
// CHECK:           ^bb0(%[[INDV:.*]]: !torch.int, %[[IT:.*]]: !torch.optional<!torch.tensor>):
// CHECK:             %[[NONE:.*]] = torch.prim.unchecked_cast %[[IT]] : !torch.optional<!torch.tensor> -> !torch.none
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.tensor>
// CHECK:             %[[COND:.*]] = torch.aten.__isnot__ %[[NONE]], %[[ARG_NONE]] : !torch.none, !torch.none -> !torch.bool
// CHECK:             torch.prim.Loop.condition %[[COND]], iter(%[[OPTIONAL]] : !torch.optional<!torch.tensor>)
// CHECK:           } : (!torch.int, !torch.bool, !torch.optional<!torch.tensor>) -> !torch.optional<!torch.tensor>
// CHECK:           %[[NONE:.*]] = torch.prim.unchecked_cast %[[LOOP_RET:.*]] : !torch.optional<!torch.tensor> -> !torch.none
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<!torch.tensor>
// CHECK:           return %[[OPTIONAL]] : !torch.optional<!torch.tensor>

func @prim.loop$region_arg_to_internal(%none: !torch.none) -> !torch.optional<!torch.tensor> {
  %int10 = torch.constant.int 10
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %optional = torch.derefine %none: !torch.none to !torch.optional<!torch.tensor>
  %ret = torch.prim.Loop %int10, %true, init(%optional)  {
  ^bb0(%arg2: !torch.int, %arg3: !torch.optional<!torch.tensor>):  // no predecessors
    %cond = torch.aten.__isnot__ %arg3, %none : !torch.optional<!torch.tensor>, !torch.none -> !torch.bool
    torch.prim.Loop.condition %cond, iter(%arg3: !torch.optional<!torch.tensor>)
  } : (!torch.int, !torch.bool, !torch.optional<!torch.tensor>) -> (!torch.optional<!torch.tensor>)
  return %ret: !torch.optional<!torch.tensor>
}
