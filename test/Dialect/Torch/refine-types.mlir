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

// -----

// The data-flow analysis does not always propagate information to the entire graph.
// This results in some lattice elements being uninitialized, which must be properly
// handled when using the lattice elements to rewrite the graph.
// In this particular case, the presence of the loop causes `torch.copy.to_vtensor`
// to end up with an uninitialized lattice element. This is the simplest graph I was
// able to come up with that reproduces such behavior.

// CHECK-LABEL:   func.func @uninitialized_lattice_elements(
// CHECK:           %{{.*}} = torch.copy.to_vtensor %{{.*}} : !torch.vtensor<*,f32>

func.func @uninitialized_lattice_elements(%arg0: !torch.vtensor<*,f32>, %arg3: !torch.tensor) -> !torch.vtensor<*,f32> {
  %true = torch.constant.bool true
  %1 = torch.constant.int 0
  %2 = torch.prim.Loop %1, %true, init(%arg3) {
  ^bb0(%arg1: !torch.int, %arg2: !torch.tensor):
    torch.prim.Loop.condition %true, iter(%arg2 : !torch.tensor)
  } : (!torch.int, !torch.bool, !torch.tensor) -> !torch.tensor
  %3 = torch.tensor_static_info_cast %2 : !torch.tensor to !torch.tensor<*,f32>
  %4 = torch.copy.to_vtensor %3 : !torch.vtensor<*,f32>
  return %4 : !torch.vtensor<*,f32>
}
