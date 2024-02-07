// RUN: torch-mlir-opt -torch-simplify-shape-calculations -split-input-file %s | FileCheck %s


// CHECK-LABEL:   func.func @refine_shape_calculate_result$basic(
// CHECK-SAME:                                              %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                              %[[ARG1:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate {
// CHECK:             %[[REFINED:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor to !torch.vtensor<[2,?],unk>
// CHECK:             torch.shape.calculate.yield %[[REFINED]] : !torch.vtensor<[2,?],unk>
// CHECK:           } shapes {
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %[[INT2]], %[[ARG1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
// CHECK:           } : !torch.vtensor<[2,?],unk>
// CHECK:           %[[RESULT_ERASED:.*]] = torch.tensor_static_info_cast %[[RESULT:.*]] : !torch.vtensor<[2,?],unk> to !torch.vtensor
// CHECK:           return %[[RESULT_ERASED]] : !torch.vtensor
func.func @refine_shape_calculate_result$basic(%arg0: !torch.vtensor, %arg1: !torch.int) -> !torch.vtensor {
  %int2 = torch.constant.int 2
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct %int2, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @refine_shape_calculate_result$clobber_one_element(
// CHECK:           %[[RESULT_ERASED:.*]] = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[?,2],unk> to !torch.vtensor
// CHECK:           return %[[RESULT_ERASED]] : !torch.vtensor
func.func @refine_shape_calculate_result$clobber_one_element(%arg0: !torch.vtensor, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.prim.If %arg2 -> () {
      // Clobber element 0 of the list. So we can only know that the result is [?,2] instead of [2,2].
      %2 = torch.aten._set_item.t %1, %int0, %arg1 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @refine_shape_calculate_result$clobber_all_elements(
// CHECK:           %[[RESULT_ERASED:.*]] = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[?,?],unk> to !torch.vtensor
// CHECK:           return %[[RESULT_ERASED]] : !torch.vtensor
func.func @refine_shape_calculate_result$clobber_all_elements(%arg0: !torch.vtensor, %arg1: !torch.int, %arg2: !torch.bool) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.prim.If %arg2 -> () {
      // Set an unknown element of the list. This clobbers our knowledge of the whole contents of the list.
      // So we can only know that the result is [?,?] instead of [2,2].
      %2 = torch.aten._set_item.t %1, %arg1, %int0 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
      torch.prim.If.yield
    } else {
      torch.prim.If.yield
    }
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// Make sure that information previously in the IR is not lost.
// CHECK-LABEL:   func.func @refine_shape_calculate_result$meet_with_existing_information(
// CHECK:           %[[RESULT_ERASED:.*]] = torch.tensor_static_info_cast %{{.*}} : !torch.vtensor<[2,3],f32> to !torch.vtensor<[?,3],f32>
// CHECK:           return %[[RESULT_ERASED]] : !torch.vtensor<[?,3],f32>
func.func @refine_shape_calculate_result$meet_with_existing_information(%arg0: !torch.vtensor<[?,3],f32>, %arg1: !torch.int) -> !torch.vtensor<[?,3],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor<[?,3],f32>
  } shapes {
    %1 = torch.prim.ListConstruct %int2, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor<[?,3],f32>
  return %0 : !torch.vtensor<[?,3],f32>
}

// Don't insert static info casts if not needed.
// CHECK-LABEL:   func.func @refine_shape_calculate_result$user_allows_type_refinement(
// CHECK-NOT:       torch.tensor_static_info_cast
func.func @refine_shape_calculate_result$user_allows_type_refinement(%arg0: !torch.vtensor) -> !torch.vtensor {
  %int2 = torch.constant.int 2
  %0 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
  %1 = torch.shape.calculate {
    torch.shape.calculate.yield %0 : !torch.vtensor
  } shapes {
    %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    torch.shape.calculate.yield.shapes %2 : !torch.list<int>
  } : !torch.vtensor
  %2 = torch.aten.tanh %1 : !torch.vtensor -> !torch.vtensor
  return %2 : !torch.vtensor
}

// CHECK-LABEL:   func.func @fully_unroll_prim_loop$unroll(
// CHECK-SAME:                                 %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                 %[[ARG1:.*]]: !torch.list<int>) -> !torch.vtensor {
// CHECK-DAG:       %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG:       %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:       %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate {
// CHECK:             torch.shape.calculate.yield %[[ARG0]] : !torch.vtensor
// CHECK:           } shapes {
// CHECK:             torch.prim.Print(%[[INT0]], %[[INT0]]) : !torch.int, !torch.int
// CHECK:             torch.prim.Print(%[[INT1]], %[[INT0]]) : !torch.int, !torch.int
// CHECK:             torch.prim.Print(%[[INT2]], %[[INT0]]) : !torch.int, !torch.int
// CHECK:             torch.shape.calculate.yield.shapes %[[ARG1]] : !torch.list<int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func.func @fully_unroll_prim_loop$unroll(%arg0: !torch.vtensor, %arg1: !torch.list<int>) -> !torch.vtensor {
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    torch.prim.Loop %int3, %true, init(%int0) {
    ^bb0(%arg2: !torch.int, %arg3: !torch.int):
      torch.prim.Print(%arg2, %arg3) : !torch.int, !torch.int
      torch.prim.Loop.condition %true, iter(%arg3: !torch.int)
    } : (!torch.int, !torch.bool, !torch.int) -> (!torch.int)
    torch.shape.calculate.yield.shapes %arg1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @fully_unroll_prim_loop$no_unroll(
// CHECK:           torch.prim.Loop
func.func @fully_unroll_prim_loop$no_unroll(%arg0: !torch.vtensor, %arg1: !torch.list<int>, %arg2: !torch.int) -> !torch.vtensor {
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    torch.prim.Loop %arg2, %true, init() {
    ^bb0(%arg3: !torch.int):
      torch.prim.Print(%arg2) : !torch.int
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    torch.shape.calculate.yield.shapes %arg1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$basic(
// CHECK-SAME:                                              %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                              %[[ARG1:.*]]: !torch.int,
// CHECK-SAME:                                              %[[ARG2:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %[[ARG1]], %[[ARG2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
func.func @abstractly_interpret_list_ops$basic(%arg0: !torch.vtensor, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor {
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct : () -> !torch.list<int>
    %2 = torch.aten.append.t %1, %arg1 : !torch.list<int>, !torch.int -> !torch.list<int>
    %3 = torch.aten.append.t %1, %arg2 : !torch.list<int>, !torch.int -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// Test the different supported mutation ops.
// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$mutation_ops(
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %int1, %arg1, %arg2, %arg3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
func.func @abstractly_interpret_list_ops$mutation_ops(%arg0: !torch.vtensor, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.aten._set_item.t %1, %int1, %arg1 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    %3 = torch.aten.append.t %1, %arg2 : !torch.list<int>, !torch.int -> !torch.list<int>
    torch.aten.insert.t %1, %int3, %arg3 : !torch.list<int>, !torch.int, !torch.int
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// Test negative indexes with set_item op.
// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$neg_index_set_item(
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %arg1, %arg2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
func.func @abstractly_interpret_list_ops$neg_index_set_item(%arg0: !torch.vtensor, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor {
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %int-2 = torch.constant.int -2
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.aten._set_item.t %1, %int-1, %arg2 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    %3 = torch.aten._set_item.t %1, %int-2, %arg1 : !torch.list<int>, !torch.int, !torch.int -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// Test interspersed mutation and evaluation ops.
// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$mix_mutation_and_evaluation_ops(
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
func.func @abstractly_interpret_list_ops$mix_mutation_and_evaluation_ops(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct : () -> !torch.list<int>
    %2 = torch.aten.len.t %1 : !torch.list<int> -> !torch.int
    %3 = torch.aten.append.t %1, %2 : !torch.list<int>, !torch.int -> !torch.list<int>
    %4 = torch.aten.len.t %1 : !torch.list<int> -> !torch.int
    %5 = torch.aten.append.t %1, %4 : !torch.list<int>, !torch.int -> !torch.list<int>
    %6 = torch.aten.len.t %1 : !torch.list<int> -> !torch.int
    %7 = torch.aten.append.t %1, %6 : !torch.list<int>, !torch.int -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$use_of_alias$not_yet_handled(
// CHECK:           torch.aten.append.t
// CHECK:           torch.aten.append.t
func.func @abstractly_interpret_list_ops$use_of_alias$not_yet_handled(%arg0: !torch.vtensor, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor {
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct : () -> !torch.list<int>
    %2 = torch.aten.append.t %1, %arg1 : !torch.list<int>, !torch.int -> !torch.list<int>
    // The value of the alias %2 is printed, but we don't handle that yet.
    torch.prim.Print(%2) : !torch.list<int>
    %3 = torch.aten.append.t %1, %arg2 : !torch.list<int>, !torch.int -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$readonly_op_in_child_region(
// CHECK-SAME:                                                                    %[[VAL_0:.*]]: !torch.vtensor,
// CHECK-SAME:                                                                    %[[VAL_1:.*]]: !torch.int) -> !torch.vtensor {
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %[[INT3]] : (!torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
func.func @abstractly_interpret_list_ops$readonly_op_in_child_region(%arg0: !torch.vtensor, %arg1: !torch.int) -> !torch.vtensor {
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int0 = torch.constant.int 0
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct : () -> !torch.list<int>
    // This readonly op in a loop doesn't block us from abstractly interpreting
    // the whole block.
    torch.prim.Loop %arg1, %true, init() {
    ^bb0(%arg3: !torch.int):
      %2 = torch.aten.__getitem__.t %1, %int0 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Print(%2) : !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %2 = torch.aten.append.t %1, %int3 : !torch.list<int>, !torch.int -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// The mutation in the child region prevents us from abstractly interpreting.
// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$mutation_in_child_region(
// CHECK:             torch.aten.append.t
func.func @abstractly_interpret_list_ops$mutation_in_child_region(%arg0: !torch.vtensor, %arg1: !torch.int) -> !torch.vtensor {
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int0 = torch.constant.int 0
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct : () -> !torch.list<int>
    torch.prim.Loop %arg1, %true, init() {
    ^bb0(%arg3: !torch.int):
      %2 = torch.aten.__getitem__.t %1, %int0 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Print(%2) : !torch.list<int>
      // This mutation prevents us from abstractly interpreting.
      %3 = torch.aten.append.t %1, %arg1 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    %2 = torch.aten.append.t %1, %int3 : !torch.list<int>, !torch.int -> !torch.list<int>
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @abstractly_interpret_list_ops$miscompile$list_identity(
// CHECK-SAME:                                                                 %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                                                 %[[ARG1:.*]]: !torch.list<int>,
// CHECK-SAME:                                                                 %[[ARG2:.*]]: !torch.bool) -> !torch.vtensor {
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[VAL_4:.*]] = torch.shape.calculate {
// CHECK:             %[[VAL_5:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor to !torch.vtensor<[3,3],unk>
// CHECK:             torch.shape.calculate.yield %[[VAL_5]] : !torch.vtensor<[3,3],unk>
// CHECK:           } shapes {
                      // Notice this torch.prim.ListConstruct....
// CHECK:             %[[VAL_6:.*]] = torch.prim.ListConstruct %[[INT3]] : (!torch.int) -> !torch.list<int>
// CHECK:             %[[VAL_7:.*]] = torch.prim.If %[[ARG2]] -> (!torch.list<int>) {
// CHECK:               torch.prim.If.yield %[[VAL_6]] : !torch.list<int>
// CHECK:             } else {
// CHECK:               torch.prim.If.yield %[[ARG1]] : !torch.list<int>
// CHECK:             }
                      // .... and this one don't have the same object identity, but should!
// CHECK:             %[[VAL_8:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             %[[VAL_9:.*]] = torch.prim.If %[[ARG2]] -> (!torch.list<int>) {
// CHECK:               torch.prim.If.yield %[[VAL_8]] : !torch.list<int>
// CHECK:             } else {
// CHECK:               torch.prim.If.yield %[[ARG1]] : !torch.list<int>
// CHECK:             }
// CHECK:             %[[VAL_10:.*]] = torch.aten.__is__ %[[VAL_11:.*]], %[[VAL_12:.*]] : !torch.list<int>, !torch.list<int> -> !torch.bool
// CHECK:             torch.prim.Print(%[[VAL_10]]) : !torch.bool
// CHECK:             torch.shape.calculate.yield.shapes %[[VAL_8]] : !torch.list<int>
// CHECK:           } : !torch.vtensor<[3,3],unk>
// CHECK:           %[[VAL_13:.*]] = torch.tensor_static_info_cast %[[VAL_14:.*]] : !torch.vtensor<[3,3],unk> to !torch.vtensor
// CHECK:           return %[[VAL_13]] : !torch.vtensor
func.func @abstractly_interpret_list_ops$miscompile$list_identity(%arg0: !torch.vtensor, %arg1: !torch.list<int>, %arg2: !torch.bool) -> !torch.vtensor {
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int0 = torch.constant.int 0
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct : () -> !torch.list<int>
    %2 = torch.aten.append.t %1, %int3 : !torch.list<int>, !torch.int -> !torch.list<int>
    // TODO: Fix this miscompile!
    // For the case where %arg2 is true, the resulting IR will miscompile
    // because the abstract interpretation of the list ops will create two list
    // literals.
    // One possible solution would be to know that torch.prim.If.yield creates
    // a new SSA name for the same dynamic value (it's not the only thing that
    // can do this -- pushing and popping the list onto another list could
    // create the same situation). Another possible solution would be to only
    // replace a single list literal at a time, and bail out if there are any
    // uses of the original list value that are not replaced by the created
    // literal.
    %3 = torch.prim.If %arg2 -> (!torch.list<int>) {
      torch.prim.If.yield %1 : !torch.list<int>
    } else {
      torch.prim.If.yield %arg1 : !torch.list<int>
    }
    %4 = torch.aten.append.t %1, %int3 : !torch.list<int>, !torch.int -> !torch.list<int>
    %5 = torch.prim.If %arg2 -> (!torch.list<int>) {
      torch.prim.If.yield %1 : !torch.list<int>
    } else {
      torch.prim.If.yield %arg1 : !torch.list<int>
    }
    %6 = torch.aten.__is__ %3, %5 : !torch.list<int>, !torch.list<int> -> !torch.bool
    torch.prim.Print(%6) : !torch.bool
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}



// "Integration test" for basic case of all the patterns working together.
// This test should usually not be the one to catch an issue.
// If it does catch an issue then it indicates a more precise unit test that is
// missing.
// CHECK-LABEL:   func.func @basic_integration(
// CHECK-SAME:                %[[ARG0:.*]]: !torch.vtensor<[?,?],unk>) -> !torch.vtensor {
// CHECK-DAG:       %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate {
// CHECK:             %[[TANH:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<[?,?],unk> -> !torch.vtensor<[?,?],unk>
// CHECK:             torch.shape.calculate.yield %[[TANH]] : !torch.vtensor<[?,?],unk>
// CHECK:           } shapes {
// CHECK:             %[[SIZE0:.*]] = torch.aten.size.int %[[ARG0]], %[[INT0]] : !torch.vtensor<[?,?],unk>, !torch.int -> !torch.int
// CHECK:             %[[SIZE1:.*]] = torch.aten.size.int %[[ARG0]], %[[INT1]] : !torch.vtensor<[?,?],unk>, !torch.int -> !torch.int
// CHECK:             %[[SHAPE:.*]] = torch.prim.ListConstruct %[[SIZE0]], %[[SIZE1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[SHAPE]] : !torch.list<int>
// CHECK:           } : !torch.vtensor<[?,?],unk>
// CHECK:           %[[RESULT_ERASED:.*]] = torch.tensor_static_info_cast %[[RESULT:.*]] : !torch.vtensor<[?,?],unk> to !torch.vtensor
// CHECK:           return %[[RESULT_ERASED]] : !torch.vtensor
func.func @basic_integration(%arg0: !torch.vtensor<[?,?],unk>) -> !torch.vtensor {
  %true = torch.constant.bool true
  %0 = torch.shape.calculate {
    %1 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?],unk> -> !torch.vtensor
    torch.shape.calculate.yield %1 : !torch.vtensor
  } shapes {
    %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %2 = torch.aten.dim %arg0 : !torch.vtensor<[?,?],unk> -> !torch.int
    torch.prim.Loop %2, %true, init() {
    ^bb0(%arg1: !torch.int):
      %3 = torch.aten.size.int %arg0, %arg1 : !torch.vtensor<[?,?],unk>, !torch.int -> !torch.int
      %4 = torch.aten.append.t %1, %3 : !torch.list<int>, !torch.int -> !torch.list<int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    torch.shape.calculate.yield.shapes %1 : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL:   func.func @fold_prim_unchecked_cast_op(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor,
// CHECK-SAME:                                           %[[VAL_1:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.vtensor {
// CHECK-DAG:       %[[VAL_2:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.shape.calculate {
// CHECK:             %[[VAL_5:.*]] = torch.tensor_static_info_cast %[[VAL_0]] : !torch.vtensor to !torch.vtensor<[?,?],unk>
// CHECK:             torch.shape.calculate.yield %[[VAL_5]] : !torch.vtensor<[?,?],unk>
// CHECK:           } shapes {
// CHECK:             %[[VAL_6:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_2]] : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
// CHECK:             %[[VAL_7:.*]] = torch.aten.size.int %[[VAL_1]], %[[VAL_3]] : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
// CHECK:             %[[VAL_8:.*]] = torch.prim.ListConstruct %[[VAL_6]], %[[VAL_7]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:             torch.shape.calculate.yield.shapes %[[VAL_8]] : !torch.list<int>
// CHECK:           } : !torch.vtensor<[?,?],unk>
// CHECK:           %[[VAL_9:.*]] = torch.tensor_static_info_cast %[[VAL_10:.*]] : !torch.vtensor<[?,?],unk> to !torch.vtensor
// CHECK:           return %[[VAL_9]] : !torch.vtensor
// CHECK:         }
func.func @fold_prim_unchecked_cast_op(%arg0: !torch.vtensor, %arg1: !torch.vtensor<[?,?],si64>) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %tensor_list = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[?,?],si64>) -> !torch.list<optional<vtensor>>
  %0 = torch.shape.calculate {
    torch.shape.calculate.yield %arg0 : !torch.vtensor
  } shapes {
    %getitem = torch.aten.__getitem__.t %tensor_list, %int0 : !torch.list<optional<vtensor>>, !torch.int -> !torch.optional<vtensor>
    %unchecked_cast = torch.prim.unchecked_cast %getitem : !torch.optional<vtensor> -> !torch.vtensor
    %size = torch.aten.size %unchecked_cast : !torch.vtensor -> !torch.list<int>
    torch.shape.calculate.yield.shapes %size : !torch.list<int>
  } : !torch.vtensor
  return %0 : !torch.vtensor
}

// CHECK-LABEL: func.func @shape_calc_with_two_uses(
// CHECK-SAME:      %[[ARG:.*]]: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
// CHECK:         %[[SHAPE_LIST:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>

// CHECK:         %[[CAST_0:.*]] = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2],f32> to !torch.vtensor
// CHECK:         %[[SHAPE_CALC_0:.*]] = torch.shape.calculate {
// CHECK:           %[[NEG_0:.*]] = torch.aten.neg %[[CAST_0]] : !torch.vtensor -> !torch.tensor<[],unk>
// CHECK:           torch.shape.calculate.yield %[[NEG_0]] : !torch.tensor<[],unk>
// CHECK:         } shapes {
// CHECK:           torch.shape.calculate.yield.shapes %[[SHAPE_LIST]] : !torch.list<int>
// CHECK:         } : !torch.tensor<[],unk>
// CHECK:         %[[CAST_1:.*]] = torch.tensor_static_info_cast %[[SHAPE_CALC_0]] : !torch.tensor<[],unk> to !torch.tensor

// CHECK:         %[[VALUE_TENSOR:.*]] = torch.copy.to_vtensor %[[CAST_1]] : !torch.vtensor
// CHECK:         %[[SHAPE_CALC_1:.*]] = torch.shape.calculate {
// CHECK:           %[[NEG_1:.*]] = torch.aten.neg %[[VALUE_TENSOR]] : !torch.vtensor -> !torch.vtensor<[],unk>
// CHECK:           torch.shape.calculate.yield %[[NEG_1]] : !torch.vtensor<[],unk>
// CHECK:         } shapes {
// CHECK:           torch.shape.calculate.yield.shapes %[[SHAPE_LIST]] : !torch.list<int>
// CHECK:         } : !torch.vtensor<[],unk>

// CHECK:         %[[CAST_2:.*]] = torch.tensor_static_info_cast %[[SHAPE_CALC_1]] : !torch.vtensor<[],unk> to !torch.vtensor
// CHECK:         torch.overwrite.tensor.contents %[[CAST_2]] overwrites %[[CAST_1]] : !torch.vtensor, !torch.tensor
// CHECK:         return %[[ARG]] : !torch.vtensor<[2],f32>
// CHECK:       }
func.func @shape_calc_with_two_uses(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
  %shape_list = torch.prim.ListConstruct : () -> !torch.list<int>

  // Negate the input tensor once.
  %tensor_0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[2],f32> to !torch.vtensor
  %shape_calc_0 = torch.shape.calculate {
    %neg_0 = torch.aten.neg %tensor_0 : !torch.vtensor -> !torch.tensor
    torch.shape.calculate.yield %neg_0 : !torch.tensor
  } shapes {
    torch.shape.calculate.yield.shapes %shape_list : !torch.list<int>
  } : !torch.tensor

  // First use of the negated tensor (to negate it again).
  %tensor_1 = torch.copy.to_vtensor %shape_calc_0 : !torch.vtensor
  %shape_calc_1 = torch.shape.calculate {
    %neg_1 = torch.aten.neg %tensor_1 : !torch.vtensor -> !torch.vtensor
    torch.shape.calculate.yield %neg_1 : !torch.vtensor
  } shapes {
    torch.shape.calculate.yield.shapes %shape_list : !torch.list<int>
  } : !torch.vtensor

  // Second use of the negated tensor.
  torch.overwrite.tensor.contents %shape_calc_1 overwrites %shape_calc_0 : !torch.vtensor, !torch.tensor

  return %arg0 : !torch.vtensor<[2],f32>
}
