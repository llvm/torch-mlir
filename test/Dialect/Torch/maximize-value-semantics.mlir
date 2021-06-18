// RUN: npcomp-opt -split-input-file -allow-unregistered-dialect %s -torch-maximize-value-semantics | FileCheck %s

// CHECK-LABEL:   func @torch.copy.tensor$basic(
// CHECK-SAME:                                  %[[ARG0:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[ARG0]], %[[ARG0]] : !torch.vtensor, !torch.vtensor
func @torch.copy.tensor$basic(%arg0: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %1 = torch.copy.to_vtensor %0 : !torch.vtensor
  %2 = torch.copy.to_vtensor %0 : !torch.vtensor
  return %1, %2 : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func @one_mutation_in_a_block(
// CHECK-SAME:                                  %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                                  %[[ARG1:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[ARG0]], %[[ARG1]] : !torch.vtensor, !torch.vtensor
func @one_mutation_in_a_block(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor) {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  %equal_to_arg0 = torch.copy.to_vtensor %0 : !torch.vtensor
  torch.overwrite.tensor %arg1 overwrites %0 : !torch.vtensor, !torch.tensor
  %equal_to_arg1 = torch.copy.to_vtensor %0 : !torch.vtensor
  return %equal_to_arg0, %equal_to_arg1 : !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func @multiple_mutations_in_a_block(
// CHECK-SAME:                                        %[[ARG0:.*]]: !torch.vtensor, %[[ARG1:.*]]: !torch.vtensor,
// CHECK-SAME:                                        %[[ARG2:.*]]: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor) {
// CHECK:           return %[[ARG0]], %[[ARG1]], %[[ARG1]], %[[ARG2]] : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor
func @multiple_mutations_in_a_block(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %arg2: !torch.vtensor) -> (!torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor) {
  // The mutable tensor we are overwriting.
  %tensor = torch.copy.to_tensor %arg0 : !torch.tensor

  // The original value.
  %equal_to_arg0 = torch.copy.to_vtensor %tensor : !torch.vtensor

  // Overwrite with %arg1
  torch.overwrite.tensor %arg1 overwrites %tensor : !torch.vtensor, !torch.tensor
  %equal_to_arg1 = torch.copy.to_vtensor %tensor : !torch.vtensor
  %equal_to_arg1_again = torch.copy.to_vtensor %tensor : !torch.vtensor

  // Overwrite with %arg2
  torch.overwrite.tensor %arg2 overwrites %tensor : !torch.vtensor, !torch.tensor
  %equal_to_arg2 = torch.copy.to_vtensor %tensor : !torch.vtensor

  return %equal_to_arg0, %equal_to_arg1, %equal_to_arg1_again, %equal_to_arg2 : !torch.vtensor, !torch.vtensor, !torch.vtensor, !torch.vtensor
}

// CHECK-LABEL:   func @unmodeled_mutation(
// CHECK:           torch.overwrite.tensor
func @unmodeled_mutation(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.copy.to_tensor %arg0 : !torch.tensor
  torch.overwrite.tensor %arg1 overwrites %0 : !torch.vtensor, !torch.tensor
  "some.op"(%0) : (!torch.tensor) -> ()
  %result = torch.copy.to_vtensor %0 : !torch.vtensor
  return %result : !torch.vtensor
}

// We don't yet handle nontrivial cases involving control flow.
// CHECK-LABEL:   func @unimplemented_control_flow(
// CHECK:           torch.copy.to_vtensor
func @unimplemented_control_flow(%arg0: !torch.vtensor, %arg1: !torch.vtensor, %cond: !torch.bool) -> (!torch.vtensor, !torch.vtensor) {
  %tensor = torch.copy.to_tensor %arg0 : !torch.tensor
  %equal_to_arg0 = torch.copy.to_vtensor %tensor : !torch.vtensor
  torch.prim.If %cond -> () {
    torch.overwrite.tensor %arg1 overwrites %tensor : !torch.vtensor, !torch.tensor
    torch.prim.If.yield
  } else {
    torch.prim.If.yield
  }
  %equal_to_arg1 = torch.copy.to_vtensor %tensor : !torch.vtensor
  return %equal_to_arg0, %equal_to_arg1 : !torch.vtensor, !torch.vtensor
}
