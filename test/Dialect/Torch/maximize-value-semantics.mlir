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

// CHECK-LABEL:   func.func @multiple_aliases_independent(
// CHECK-SAME:                                    %[[ARG0:.*]]: !torch.vtensor) -> (!torch.tensor, !torch.tensor) {
// CHECK:           %[[SLICE1:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[INT0:.*]], %[[INT0]], %[[INT2:.*]], %[[INT1:.*]] : !torch.vtensor, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor
// CHECK:           %[[SLICE2:.*]] = torch.aten.slice.Tensor %[[ARG0]], %[[INT0]], %[[INT2]], %[[INT4:.*]], %[[INT1]] : !torch.vtensor, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor 
func.func @multiple_aliases_independent(%arg0: !torch.vtensor) -> (!torch.tensor, !torch.tensor) {
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %0 = torch.tensor_static_info_cast %arg0 : !torch.vtensor to !torch.vtensor
  %1 = torch.copy.to_tensor %0 : !torch.tensor
  %2 = torch.aten.slice.Tensor %1, %int0, %int0, %int2, %int1 : !torch.tensor, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor
  %3 = torch.aten.slice.Tensor %1, %int0, %int2, %int4, %int1 : !torch.tensor, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor
  %4 = torch.copy.to_vtensor %2 : !torch.vtensor
  %5 = torch.aten.fill.Scalar %4, %int1 : !torch.vtensor, !torch.int -> !torch.vtensor
  torch.overwrite.tensor.contents %5 overwrites %2 : !torch.vtensor, !torch.tensor
  %6 = torch.copy.to_vtensor %3 : !torch.vtensor
  %7 = torch.aten.fill.Scalar %6, %int2 : !torch.vtensor, !torch.int -> !torch.vtensor
  torch.overwrite.tensor.contents %7 overwrites %3 : !torch.vtensor, !torch.tensor
  return %2, %3 : !torch.tensor, !torch.tensor
}

// CHECK-LABEL:  func.func @multiple_aliases_independent_lstm_cell(
// CHECK-SAME:                                   %[[ARG0:.*]]: !torch.vtensor<[2,10],f32>) -> !torch.vtensor<[2,20],f32> {
// CHECK:       %[[SLICE1:.*]] = torch.aten.slice.Tensor %[[tensor:.*]], %[[INT1:.*]], %[[INT0:.*]], %[[INT20:.*]], %[[INT1]] : !torch.vtensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,20],f32>
// CHECK:       %[[SLICE2:.*]] = torch.aten.slice.Tensor %[[tensor]], %[[INT1]], %[[INT20]], %[[INT40:.*]], %[[INT1]] : !torch.vtensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,20],f32>
// CHECK:       %[[SLICE3:.*]] = torch.aten.slice.Tensor %[[tensor]], %[[INT1]], %[[INT40]], %[[INT60:.*]], %[[INT1]] : !torch.vtensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,20],f32>
// CHECK:       %[[SLICE4:.*]] = torch.aten.slice.Tensor %[[tensor]], %[[INT1]], %[[INT60]], %[[INT80:.*]], %[[INT1]] : !torch.vtensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,20],f32>
func.func @multiple_aliases_independent_lstm_cell(%arg0: !torch.vtensor<[2,10],f32>) -> !torch.vtensor<[2,20],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.vtensor.literal(dense_resource<__elided__> : tensor<80x20xf32>) : !torch.vtensor<[80,20],f32>
  %1 = torch.vtensor.literal(dense_resource<__elided__> : tensor<80x10xf32>) : !torch.vtensor<[80,10],f32>
  %2 = torch.vtensor.literal(dense_resource<__elided__> : tensor<80xf32>) : !torch.vtensor<[80],f32>
  %int2 = torch.constant.int 2
  %int20 = torch.constant.int 20
  %int6 = torch.constant.int 6
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int40 = torch.constant.int 40
  %int60 = torch.constant.int 60
  %int80 = torch.constant.int 80
  %3 = torch.prim.ListConstruct %int2, %int20 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %4 = torch.aten.zeros %3, %int6, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2,20],f32>
  %5 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[80,20],f32>, !torch.int, !torch.int -> !torch.vtensor<[20,80],f32>
  %6 = torch.aten.mm %4, %5 : !torch.vtensor<[2,20],f32>, !torch.vtensor<[20,80],f32> -> !torch.vtensor<[2,80],f32>
  %7 = torch.aten.mul.Scalar %2, %int1 : !torch.vtensor<[80],f32>, !torch.int -> !torch.vtensor<[80],f32>
  %8 = torch.aten.add.Tensor %7, %6, %int1 : !torch.vtensor<[80],f32>, !torch.vtensor<[2,80],f32>, !torch.int -> !torch.vtensor<[2,80],f32>
  %9 = torch.copy.to_tensor %8 : !torch.tensor<[2,80],f32>
  %10 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[80,10],f32>, !torch.int, !torch.int -> !torch.vtensor<[10,80],f32>
  %11 = torch.aten.mm %arg0, %10 : !torch.vtensor<[2,10],f32>, !torch.vtensor<[10,80],f32> -> !torch.vtensor<[2,80],f32>
  %12 = torch.aten.mul.Scalar %2, %int1 : !torch.vtensor<[80],f32>, !torch.int -> !torch.vtensor<[80],f32>
  %13 = torch.aten.add.Tensor %12, %11, %int1 : !torch.vtensor<[80],f32>, !torch.vtensor<[2,80],f32>, !torch.int -> !torch.vtensor<[2,80],f32>
  %14 = torch.copy.to_vtensor %9 : !torch.vtensor<[2,80],f32>
  %15 = torch.aten.add.Tensor %14, %13, %int1 : !torch.vtensor<[2,80],f32>, !torch.vtensor<[2,80],f32>, !torch.int -> !torch.vtensor<[2,80],f32>
  torch.overwrite.tensor.contents %15 overwrites %9 : !torch.vtensor<[2,80],f32>, !torch.tensor<[2,80],f32>
  %16 = torch.aten.slice.Tensor %9, %int1, %int0, %int20, %int1 : !torch.tensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor<[2,20],f32>
  %17 = torch.aten.slice.Tensor %9, %int1, %int20, %int40, %int1 : !torch.tensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor<[2,20],f32>
  %18 = torch.aten.slice.Tensor %9, %int1, %int40, %int60, %int1 : !torch.tensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor<[2,20],f32>
  %19 = torch.aten.slice.Tensor %9, %int1, %int60, %int80, %int1 : !torch.tensor<[2,80],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.tensor<[2,20],f32>
  %20 = torch.copy.to_vtensor %16 : !torch.vtensor<[2,20],f32>
  %21 = torch.aten.sigmoid %20 : !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  torch.overwrite.tensor.contents %21 overwrites %16 : !torch.vtensor<[2,20],f32>, !torch.tensor<[2,20],f32>
  %22 = torch.copy.to_vtensor %17 : !torch.vtensor<[2,20],f32>
  %23 = torch.aten.sigmoid %22 : !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  torch.overwrite.tensor.contents %23 overwrites %17 : !torch.vtensor<[2,20],f32>, !torch.tensor<[2,20],f32>
  %24 = torch.copy.to_vtensor %18 : !torch.vtensor<[2,20],f32>
  %25 = torch.aten.tanh %24 : !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  torch.overwrite.tensor.contents %25 overwrites %18 : !torch.vtensor<[2,20],f32>, !torch.tensor<[2,20],f32>
  %26 = torch.copy.to_vtensor %19 : !torch.vtensor<[2,20],f32>
  %27 = torch.aten.sigmoid %26 : !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  torch.overwrite.tensor.contents %27 overwrites %19 : !torch.vtensor<[2,20],f32>, !torch.tensor<[2,20],f32>
  %28 = torch.copy.to_vtensor %17 : !torch.vtensor<[2,20],f32>
  %29 = torch.aten.mul.Tensor %28, %4 : !torch.vtensor<[2,20],f32>, !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  %30 = torch.copy.to_vtensor %16 : !torch.vtensor<[2,20],f32>
  %31 = torch.copy.to_vtensor %18 : !torch.vtensor<[2,20],f32>
  %32 = torch.aten.mul.Tensor %30, %31 : !torch.vtensor<[2,20],f32>, !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  %33 = torch.aten.add.Tensor %29, %32, %int1 : !torch.vtensor<[2,20],f32>, !torch.vtensor<[2,20],f32>, !torch.int -> !torch.vtensor<[2,20],f32>
  %34 = torch.aten.tanh %33 : !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  %35 = torch.copy.to_vtensor %19 : !torch.vtensor<[2,20],f32>
  %36 = torch.aten.mul.Tensor %35, %34 : !torch.vtensor<[2,20],f32>, !torch.vtensor<[2,20],f32> -> !torch.vtensor<[2,20],f32>
  return %36 : !torch.vtensor<[2,20],f32>
}
