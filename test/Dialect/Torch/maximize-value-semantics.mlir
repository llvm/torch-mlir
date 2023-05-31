// RUN: torch-mlir-opt -split-input-file -allow-unregistered-dialect %s -torch-maximize-value-semantics | FileCheck %s

// CHECK-LABEL:   func.func @torch.copy.tensor$basic(
// CHECK-SAME:                                  %[[ARG0:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[ARG0]], %[[ARG0]] : !torch.vtensor, !torch.vtensor
func.func @torch.copy.tensor$basic(%arg0: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.copy.to_vtensor %0 : !torch.vtensor
  %2 = torch.copy.to_vtensor %0 : !torch.vtensor
  return %1, %2 : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func.func @one_mutation_in_a_block(
// CHECK-SAME:                                  %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                  %[[ARG1:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[ARG0]], %[[ARG1]] : !torch.vtensor, !torch.vtensor
func.func @one_mutation_in_a_block(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %equal_to_arg0 = torch.copy.to_vtensor %0 : !torch.vtensor
  torch.overwrite.tensor.contents %arg1 overwrites %0 : !torch.vtensor, !torch.tensor
  %equal_to_arg1 = torch.copy.to_vtensor %0 : !torch.vtensor
  return %equal_to_arg0, %equal_to_arg1 : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func.func @multiple_mutations_in_a_block(
// CHECK-SAME:                                        %[[ARG0:.*]]: !torch.vtensor, %[[ARG1:.*]]: !torch.vtensor,
// CHECK-SAME:                                        %[[ARG2:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[ARG0]], %[[ARG1]], %[[ARG1]], %[[ARG2]] : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor
func.func @multiple_mutations_in_a_block(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %arg2: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor) {
  // The mutable tensor we are overwriting.
  %tensor = torch.copy.to_tensor %arg0 : !torch.tensor

  // The original value.
  %equal_to_arg0 = torch.copy.to_vtensor %tensor : !torch.vtensor

  // Overwrite with %arg1
  torch.overwrite.tensor.contents %arg1 overwrites %tensor : !torch.vtensor, !torch.tensor
  %equal_to_arg1 = torch.copy.to_vtensor %tensor : !torch.vtensor
  %equal_to_arg1_again = torch.copy.to_vtensor %tensor : !torch.vtensor

  // Overwrite with %arg2
  torch.overwrite.tensor.contents %arg2 overwrites %tensor : !torch.vtensor, !torch.tensor
  %equal_to_arg2 = torch.copy.to_vtensor %tensor : !torch.vtensor

  return %equal_to_arg0, %equal_to_arg1, %equal_to_arg1_again, %equal_to_arg2 : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func.func @mutation_followed_by_view_like_ops(
// CHECK-SAME:                                             %[[VALUE_T:.*]]: !torch.vtensor, %[[OVERWRITER:.*]]: !torch.vtensor, %[[INT_LIST:.*]]: !torch.list<int>) -> !torch.vtensor {
// CHECK:           %[[VIEW:.*]] = torch.aten.view %[[OVERWRITER]], %[[INT_LIST]] : !torch.vtensor, !torch.list<int> -> !torch.vtensor
// CHECK:           %[[RESULT:.*]] = torch.aten.permute %[[VIEW]], %[[INT_LIST]] : !torch.vtensor, !torch.list<int> -> !torch.vtensor
// CHECK:           return %[[RESULT]] : !torch.vtensor
func.func @mutation_followed_by_view_like_ops(%value_t: !torch.vtensor, %overwriter: !torch.vtensor, %int_list: !torch.list<int>) -> !torch.vtensor {
  %t = torch.copy.to_tensor %value_t : !torch.tensor
  torch.overwrite.tensor.contents %overwriter overwrites %t : !torch.vtensor, !torch.tensor
  %view = torch.aten.view %t, %int_list : !torch.tensor, !torch.list<int> -> !torch.tensor
  %result = torch.aten.permute %view, %int_list : !torch.tensor, !torch.list<int> -> !torch.tensor
  %value_result = torch.copy.to_vtensor %result : !torch.vtensor
  return %value_result : !torch.vtensor
}

// CHECK-LABEL:   func.func @mutation_of_view_like_op_result(
// CHECK-SAME:                                             %[[VALUE_T:.*]]: !torch.vtensor, %[[OVERWRITER:.*]]: !torch.vtensor, %[[INT_LIST:.*]]: !torch.list<int>) -> !torch.vtensor {
// CHECK:           return %[[OVERWRITER]] : !torch.vtensor
func.func @mutation_of_view_like_op_result(%value_t: !torch.vtensor, %overwriter: !torch.vtensor, %int_list: !torch.list<int>) -> !torch.vtensor {
  %t = torch.copy.to_tensor %value_t : !torch.tensor
  %view = torch.aten.view %t, %int_list : !torch.tensor, !torch.list<int> -> !torch.tensor
  torch.overwrite.tensor.contents %overwriter overwrites %view : !torch.vtensor, !torch.tensor
  %result = torch.copy.to_vtensor %view : !torch.vtensor
  return %result : !torch.vtensor
}

// CHECK-LABEL:   func.func @value_tensor_used_after_copy_was_mutated(
// CHECK-SAME:                                                       %[[VALUE_T:.*]]: !torch.vtensor,
// CHECK-SAME:                                                       %[[OVERWRITER:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[VALUE_T]], %[[OVERWRITER]] : !torch.vtensor, !torch.vtensor
func.func @value_tensor_used_after_copy_was_mutated(%value_t: !torch.vtensor, %overwriter: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
  %t = torch.copy.to_tensor %value_t : !torch.tensor
  torch.overwrite.tensor.contents %overwriter overwrites %t : !torch.vtensor, !torch.tensor
  %value_mutated_t = torch.copy.to_vtensor %t : !torch.vtensor
  return %value_t, %value_mutated_t : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func.func @unmodeled_mutation(
// CHECK:           torch.overwrite.tensor.contents
func.func @unmodeled_mutation(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  torch.overwrite.tensor.contents %arg1 overwrites %0 : !torch.vtensor, !torch.tensor
  "some.op"(%0) : (!torch.tensor) -> ()
  %result = torch.copy.to_vtensor %0 : !torch.vtensor
  return %result : !torch.vtensor
}

// CHECK-LABEL:   func.func @control_flow$no_overwrites(
// CHECK-SAME:                                          %[[ARG0:.*]]: !torch.vtensor, %[[ARG1:.*]]: !torch.vtensor, %[[COND:.*]]: !torch.bool) -> !torch.vtensor {
// CHECK:             torch.prim.If.yield %[[ARG0]] : !torch.vtensor
// CHECK:             torch.prim.If.yield %[[ARG1]] : !torch.vtensor
func.func @control_flow$no_overwrites(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %cond: !torch.bool) -> (!torch.vtensor) {
  %tensor0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %tensor1 = torch.copy.to_tensor %arg1 : !torch.tensor
  %new_tensor = torch.prim.If %cond -> (!torch.vtensor) {
    %vtensor0 = torch.copy.to_vtensor %tensor0 : !torch.vtensor
    torch.prim.If.yield %vtensor0 : !torch.vtensor
  } else {
    %vtensor1 = torch.copy.to_vtensor %tensor1 : !torch.vtensor
    torch.prim.If.yield %vtensor1 : !torch.vtensor
  }
  return %new_tensor : !torch.vtensor
}

// We don't yet handle nontrivial cases involving control flow.
// CHECK-LABEL:   func.func @unimplemented_control_flow(
// CHECK:           torch.copy.to_vtensor
func.func @unimplemented_control_flow(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %cond: !torch.bool) -> (!torch.vtensor, !torch.vtensor) {
  %tensor = torch.copy.to_tensor %arg0 : !torch.tensor
  %equal_to_arg0 = torch.copy.to_vtensor %tensor : !torch.vtensor
  torch.prim.If %cond -> () {
    torch.overwrite.tensor.contents %arg1 overwrites %tensor : !torch.vtensor, !torch.tensor
    torch.prim.If.yield
  } else {
    torch.prim.If.yield
  }
  %equal_to_arg1 = torch.copy.to_vtensor %tensor : !torch.vtensor
  return %equal_to_arg0, %equal_to_arg1 : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func.func @non_value_tensor_returned(
// CHECK-SAME:                                    %[[VALUE_T:.*]]: !torch.vtensor) -> !torch.tensor {
// CHECK:           %[[T:.*]] = torch.copy.to_tensor %[[VALUE_T]] : !torch.tensor
// CHECK:           return %[[T]] : !torch.tensor
func.func @non_value_tensor_returned(%value_t: !torch.vtensor) -> !torch.tensor {
  %t = torch.copy.to_tensor %value_t : !torch.tensor
  return %t : !torch.tensor
}

// CHECK-LABEL:   func.func @non_value_tensor_returned$with_overwrite(
// CHECK-SAME:                                                   %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                                   %{{.*}}: !torch.vtensor) -> !torch.tensor {
// CHECK:           %[[RESULT:.*]] = torch.copy.to_tensor %[[ARG0]] : !torch.tensor
// CHECK:           return %[[RESULT]] : !torch.tensor
func.func @non_value_tensor_returned$with_overwrite(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.tensor {
  %2 = torch.copy.to_tensor %arg1 : !torch.tensor
  torch.overwrite.tensor.contents %arg0 overwrites %2 : !torch.vtensor, !torch.tensor
  return %2 : !torch.tensor
}

// CHECK-LABEL:   func.func @non_value_tensor_returned$return_from_multiple_slices(
// CHECK-SAME:                                                                %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                                                %[[ARG1:.*]]: !torch.vtensor) -> (!torch.tensor, !torch.vtensor, !torch.tensor) {
// CHECK:           %[[NON_VALUE_TENSOR0:.*]] = torch.copy.to_tensor %[[ARG0]] : !torch.tensor
// CHECK:           %[[NON_VALUE_TENSOR1:.*]] = torch.copy.to_tensor %[[ARG1]] : !torch.tensor
// CHECK:           return %[[NON_VALUE_TENSOR0]], %[[ARG0]], %[[NON_VALUE_TENSOR1]] : !torch.tensor, !torch.vtensor, !torch.tensor
func.func @non_value_tensor_returned$return_from_multiple_slices(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> (!torch.tensor, !torch.vtensor, !torch.tensor) {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  // Make a vtensor copy and return that, just to have a load-bearing use.
  // This test mainly checks the rewriting of the non-value tensor returns
  // though.
  %1 = torch.copy.to_vtensor %0 : !torch.vtensor
  %2 = torch.copy.to_tensor %arg1 : !torch.tensor
  return %0, %1, %2 : !torch.tensor, !torch.vtensor, !torch.tensor
}

// CHECK-LABEL:   func.func @viewlike$basic_unsqueeze(
// CHECK-SAME:                                   %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze %[[ARG]], %[[INT0]] : !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:           return %[[UNSQUEEZE]] : !torch.vtensor
func.func @viewlike$basic_unsqueeze(%arg0: !torch.vtensor) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.aten.unsqueeze %0, %int0 : !torch.tensor, !torch.int -> !torch.tensor
  %2 = torch.copy.to_vtensor %1 : !torch.vtensor
  return %2 : !torch.vtensor
}

// CHECK-LABEL:   func.func @viewlike$basic_flatten(
// CHECK-SAME:                                 %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[INTM1:.*]] = torch.constant.int -1
// CHECK:           %[[FLATTEN:.*]] = torch.aten.flatten.using_ints %[[ARG]], %[[INT0]], %[[INTM1]] : !torch.vtensor, !torch.int, !torch.int -> !torch.vtensor
// CHECK:           return %[[FLATTEN]] : !torch.vtensor
func.func @viewlike$basic_flatten(%arg0: !torch.vtensor) -> !torch.vtensor {
  %start = torch.constant.int 0
  %end = torch.constant.int -1
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.aten.flatten.using_ints %0, %start, %end : !torch.tensor, !torch.int, !torch.int -> !torch.tensor
  %2 = torch.copy.to_vtensor %1 : !torch.vtensor
  return %2 : !torch.vtensor
}

// CHECK-LABEL:   func.func @viewlike$transitive(
// CHECK-SAME:                              %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[UNSQUEEZE0:.*]] = torch.aten.unsqueeze %[[ARG]], %[[INT0]] : !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:           %[[UNSQUEEZE1:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE0]], %[[INT0]] : !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:           return %[[UNSQUEEZE1]] : !torch.vtensor
func.func @viewlike$transitive(%arg0: !torch.vtensor) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.aten.unsqueeze %0, %int0 : !torch.tensor, !torch.int -> !torch.tensor
  %2 = torch.aten.unsqueeze %1, %int0 : !torch.tensor, !torch.int -> !torch.tensor
  %3 = torch.copy.to_vtensor %2 : !torch.vtensor
  return %3 : !torch.vtensor
}

// CHECK-LABEL:   func.func @viewlike$transitive_tree(
// CHECK-SAME:                                   %[[ARG:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           %[[INT0:.*]] = torch.constant.int 0
// CHECK:           %[[UNSQUEEZE0:.*]] = torch.aten.unsqueeze %[[ARG]], %[[INT0]] : !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:           %[[RET0:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE0]], %[[INT0]] : !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:           %[[RET1:.*]] = torch.aten.unsqueeze %[[UNSQUEEZE0]], %[[INT0]] : !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:           return %[[RET0]], %[[RET1]] : !torch.vtensor, !torch.vtensor
func.func @viewlike$transitive_tree(%arg0: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
  %int0 = torch.constant.int 0
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  // %1 has two users.
  %1 = torch.aten.unsqueeze %0, %int0 : !torch.tensor, !torch.int -> !torch.tensor

  %2 = torch.aten.unsqueeze %1, %int0 : !torch.tensor, !torch.int -> !torch.tensor
  %3 = torch.copy.to_vtensor %2 : !torch.vtensor

  %4 = torch.aten.unsqueeze %1, %int0 : !torch.tensor, !torch.int -> !torch.tensor
  %5 = torch.copy.to_vtensor %4 : !torch.vtensor

  return %3, %5 : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func.func @viewlike$unmodeled_op(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[UNSQUEEZE:.*]] = torch.aten.unsqueeze {{.*}} : !torch.tensor, !torch.int -> !torch.tensor
// CHECK:           "some.op"(%[[UNSQUEEZE]]) : (!torch.tensor) -> ()
func.func @viewlike$unmodeled_op(%arg0: !torch.vtensor) -> !torch.vtensor {
  %int0 = torch.constant.int 0
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.aten.unsqueeze %0, %int0 : !torch.tensor, !torch.int -> !torch.tensor
  "some.op"(%1) : (!torch.tensor) -> ()
  %2 = torch.copy.to_vtensor %1 : !torch.vtensor
  return %2 : !torch.vtensor
}

// CHECK-LABEL:   func.func @viewlike$two_inputs_one_copy(
// CHECK-SAME:                                       %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[EXPAND_AS:.*]] = torch.aten.expand_as %[[ARG]], %[[ARG]] : !torch.vtensor, !torch.vtensor -> !torch.vtensor
// CHECK:           return %[[EXPAND_AS]] : !torch.vtensor
func.func @viewlike$two_inputs_one_copy(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.aten.expand_as %0, %0 : !torch.tensor, !torch.tensor -> !torch.tensor
  %2 = torch.copy.to_vtensor %1 : !torch.vtensor
  return %2 : !torch.vtensor
}

// CHECK-LABEL:   func.func @viewlike$two_inputs_two_copies(
// CHECK-SAME:                                         %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                         %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[EXPAND_AS:.*]] = torch.aten.expand_as %[[ARG0]], %[[ARG1]] : !torch.vtensor, !torch.vtensor -> !torch.vtensor
// CHECK:           return %[[EXPAND_AS]] : !torch.vtensor
func.func @viewlike$two_inputs_two_copies(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.copy.to_tensor %arg1 : !torch.tensor
  %2 = torch.aten.expand_as %0, %1 : !torch.tensor, !torch.tensor -> !torch.tensor
  %3 = torch.copy.to_vtensor %2 : !torch.vtensor
  return %3 : !torch.vtensor
}

// CHECK-LABEL:   func.func @castlike(
// CHECK-SAME:                       %[[ARG0:.*]]: !torch.vtensor<[5,4],f32>) -> !torch.tensor {
// CHECK:           %[[CAST1:.*]] = torch.tensor_static_info_cast %[[ARG0]] : !torch.vtensor<[5,4],f32> to !torch.vtensor
// CHECK:           %[[CAST2:.*]] = torch.tensor_static_info_cast %[[CAST1]] : !torch.vtensor to !torch.vtensor<[5,4],f32>
// CHECK:           %[[CAST3:.*]] = torch.tensor_static_info_cast %[[CAST2]] : !torch.vtensor<[5,4],f32> to !torch.vtensor
// CHECK:           %[[COPY:.*]] = torch.copy.to_tensor %[[CAST3]] : !torch.tensor
// CHECK:           return %[[COPY]] : !torch.tensor
func.func @castlike(%arg0: !torch.vtensor<[5,4],f32>) -> !torch.tensor {
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor<[5,4],f32> to !torch.vtensor
  %1 = torch.copy.to_tensor %0 : !torch.tensor
  %2 = torch.tensor_static_info_cast %1 : !torch.tensor to !torch.tensor<[5,4],f32>
  %3 = torch.copy.to_vtensor %2 : !torch.vtensor<[5,4],f32>
  torch.overwrite.tensor.contents %3 overwrites %2 : !torch.vtensor<[5,4],f32>, !torch.tensor<[5,4],f32>
  return %1 : !torch.tensor
}
