// RUN: torch-mlir-opt -torch-refine-types -split-input-file %s | FileCheck %s

// This file tests the structural logic of the pass. This is for testing logic
// that does not scale with the number of ops supported, such as the core
// propagation logic, rewriting, etc.
// Code for testing transfer functions for new ops (which is most changes)
// should go in refine-types-ops.mlir.

// -----

// Check that we don't crash on this input.

// CHECK-LABEL: func.func @forward
func.func @forward() -> !torch.vtensor {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct  : () -> !torch.list<tensor>
  // CHECK: torch.aten.tensor
  %1 = torch.aten.tensor %0, %none, %none, %false : !torch.list<tensor>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor
  return %1 : !torch.vtensor
}

// -----

// Check that we don't crash on this input.
// TODO: This appears to result in aten.mul.Tensor not being visited.
// We should investigate why that happens.

// CHECK-LABEL: func.func @forward
func.func @forward(%arg0: !torch.bool, %arg1: !torch.tensor) {
  %0 = torch.prim.If %arg0 -> (!torch.tensor) {
    torch.prim.If.yield %arg1 : !torch.tensor
  } else {
    torch.prim.If.yield %arg1 : !torch.tensor
  }
  %1 = torch.copy.to_vtensor %0 : !torch.vtensor
  %2 = torch.aten.mul.Tensor %1, %1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return
}
