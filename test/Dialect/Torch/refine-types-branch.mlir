// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL:   func.func @prim.if$branch_merge_type_tensor(
// CHECK-SAME:                                                   %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:                                                   %[[T1:.*]]: !torch.tensor,
// CHECK-SAME:                                                   %[[T2:.*]]: !torch.tensor) -> !torch.bool {
// CHECK:           %[[MERGED:.*]] = torch.prim.If %[[PRED]] -> (!torch.optional<tensor>) {
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[T1]] : !torch.tensor to !torch.optional<tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<tensor>
// CHECK:           } else {
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[T2]] : !torch.tensor to !torch.optional<tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<tensor>
// CHECK:           }
// CHECK:           %[[REFINED:.*]] = torch.prim.unchecked_cast %[[MERGED:.*]] : !torch.optional<tensor> -> !torch.tensor
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[RET:.*]] = torch.aten.__isnot__ %[[REFINED]], %[[NONE]] : !torch.tensor, !torch.none -> !torch.bool
// CHECK:           return %[[RET]] : !torch.bool

func.func @prim.if$branch_merge_type_tensor(%pred: !torch.bool, %t0: !torch.tensor, %t1: !torch.tensor) -> !torch.bool {
  %res = torch.prim.If %pred -> (!torch.optional<tensor>) {
    %optional0 = torch.derefine %t0: !torch.tensor to !torch.optional<tensor>
    torch.prim.If.yield %optional0: !torch.optional<tensor>
  } else {
    %optional1 = torch.derefine %t1: !torch.tensor to !torch.optional<tensor>
    torch.prim.If.yield %optional1: !torch.optional<tensor>
  }
  %none = torch.constant.none
  %cmp = torch.aten.__isnot__ %res, %none : !torch.optional<tensor>, !torch.none -> !torch.bool
  return %cmp : !torch.bool
}

// -----

// CHECK-LABEL:   func.func @prim.if$branch_merge_type_optional(
// CHECK-SAME:                                                     %[[PRED:.*]]: !torch.bool,
// CHECK-SAME:                                                     %[[T:.*]]: !torch.tensor) -> !torch.optional<tensor> {
// CHECK:           %[[MERGED:.*]] = torch.prim.If %[[PRED]] -> (!torch.optional<tensor>) {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<tensor>
// CHECK:           } else {
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[T]] : !torch.tensor to !torch.optional<tensor>
// CHECK:             torch.prim.If.yield %[[OPTIONAL]] : !torch.optional<tensor>
// CHECK:           }
// CHECK:           return %[[MERGED:.*]] : !torch.optional<tensor>

func.func @prim.if$branch_merge_type_optional(%pred: !torch.bool, %t1: !torch.tensor) -> !torch.optional<tensor> {
  %res = torch.prim.If %pred -> (!torch.optional<tensor>) {
    %none = torch.constant.none
    %optional0 = torch.derefine %none: !torch.none to !torch.optional<tensor>
    torch.prim.If.yield %optional0: !torch.optional<tensor>
  } else {
    %optional1 = torch.derefine %t1: !torch.tensor to !torch.optional<tensor>
    torch.prim.If.yield %optional1: !torch.optional<tensor>
  }
  return %res: !torch.optional<tensor>
}

// -----

// CHECK-LABEL:   func.func @prim.if$refined_type_conflicting(
// CHECK-SAME:                                                   %[[NONE:.*]]: !torch.none) -> !torch.tensor {
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<tensor>
// CHECK:           %[[NOT_NONE:.*]] = torch.aten.__isnot__ %[[NONE]], %[[NONE]] : !torch.none, !torch.none -> !torch.bool
// CHECK:           %[[PRED:.*]] = torch.prim.If %[[NOT_NONE]] -> (!torch.tensor) {
// CHECK:             %[[T:.*]] = torch.prim.unchecked_cast %[[OPTIONAL]] : !torch.optional<tensor> -> !torch.tensor
// CHECK:             torch.prim.If.yield %[[T]] : !torch.tensor
// CHECK:           } else {
// CHECK:             %[[LITERAL:.*]] = torch.tensor.literal(dense<0.000000e+00> : tensor<3x5xf32>) : !torch.tensor
// CHECK:             torch.prim.If.yield %[[LITERAL]] : !torch.tensor
// CHECK:           }
// CHECK:           return %[[PRED:.*]] : !torch.tensor

func.func @prim.if$refined_type_conflicting(%none: !torch.none) -> !torch.tensor {
  %optional = torch.derefine %none: !torch.none to !torch.optional<tensor>
  %pred = torch.aten.__isnot__ %optional, %none : !torch.optional<tensor>, !torch.none -> !torch.bool
  %res = torch.prim.If %pred -> (!torch.tensor) {
  %t = torch.prim.unchecked_cast %optional: !torch.optional<tensor> -> !torch.tensor
  torch.prim.If.yield %t: !torch.tensor
  } else {
  %t_cst = torch.tensor.literal(dense<0.0> : tensor<3x5xf32>) : !torch.tensor
  torch.prim.If.yield %t_cst: !torch.tensor
  }
  return %res: !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @prim.loop$region_arg_to_internal(
// CHECK-SAME:                            %[[ARG_NONE:.*]]: !torch.none) -> !torch.optional<tensor> {
// CHECK:           %[[INT10:.*]] = torch.constant.int 10
// CHECK:           %[[INDV:.*]] = torch.constant.int 0
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[ARG_NONE]] : !torch.none to !torch.optional<tensor>
// CHECK:           %[[LOOP_RET:.*]] = torch.prim.Loop %[[INT10]], %[[TRUE]], init(%[[OPTIONAL]])  {
// CHECK:           ^bb0(%[[INDV:.*]]: !torch.int, %[[IT:.*]]: !torch.optional<tensor>):
// CHECK:             %[[NONE:.*]] = torch.prim.unchecked_cast %[[IT]] : !torch.optional<tensor> -> !torch.none
// CHECK:             %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<tensor>
// CHECK:             %[[COND:.*]] = torch.aten.__isnot__ %[[NONE]], %[[ARG_NONE]] : !torch.none, !torch.none -> !torch.bool
// CHECK:             torch.prim.Loop.condition %[[COND]], iter(%[[OPTIONAL]] : !torch.optional<tensor>)
// CHECK:           } : (!torch.int, !torch.bool, !torch.optional<tensor>) -> !torch.optional<tensor>
// CHECK:           %[[NONE:.*]] = torch.prim.unchecked_cast %[[LOOP_RET:.*]] : !torch.optional<tensor> -> !torch.none
// CHECK:           %[[OPTIONAL:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<tensor>
// CHECK:           return %[[OPTIONAL]] : !torch.optional<tensor>

func.func @prim.loop$region_arg_to_internal(%none: !torch.none) -> !torch.optional<tensor> {
  %int10 = torch.constant.int 10
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %optional = torch.derefine %none: !torch.none to !torch.optional<tensor>
  %ret = torch.prim.Loop %int10, %true, init(%optional)  {
  ^bb0(%arg2: !torch.int, %arg3: !torch.optional<tensor>):  // no predecessors
    %cond = torch.aten.__isnot__ %arg3, %none : !torch.optional<tensor>, !torch.none -> !torch.bool
    torch.prim.Loop.condition %cond, iter(%arg3: !torch.optional<tensor>)
  } : (!torch.int, !torch.bool, !torch.optional<tensor>) -> (!torch.optional<tensor>)
  return %ret: !torch.optional<tensor>
}

// -----

// CHECK-LABEL:   func.func @f
// CHECK: %[[ATEN:.*]] = torch.aten.cos %{{.*}} : !torch.vtensor -> !torch.vtensor<*,f32>
// CHECK: %[[CAST:.*]] = torch.tensor_static_info_cast %[[ATEN]] : !torch.vtensor<*,f32> to !torch.vtensor
// CHECK: return %[[CAST]] : !torch.vtensor
func.func @f(%arg0: !torch.vtensor<*,f32>) -> !torch.vtensor {
  %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<*,f32> to !torch.vtensor
  cf.br ^bb1(%cast: !torch.vtensor)
^bb1(%arg1: !torch.vtensor):
  %1 = torch.aten.cos %arg1 : !torch.vtensor -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// CHECK-LABEL:   func.func @f
// CHECK: func.func private @callee
// CHECK-NEXT: torch.aten.cos %{{.*}} : !torch.vtensor -> !torch.vtensor<*,f32>
func.func @f() {
  builtin.module {
    func.func private @callee(%arg0: !torch.vtensor) {
      %1 = torch.aten.cos %arg0 : !torch.vtensor -> !torch.vtensor
      return
    }
    func.func @caller(%arg0: !torch.vtensor<*,f32>) {
      %cast = torch.tensor_static_info_cast %arg0 : !torch.vtensor<*,f32> to !torch.vtensor
      call @callee(%cast) : (!torch.vtensor) -> ()
      return
    }
  }
  return
}
