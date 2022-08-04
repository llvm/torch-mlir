// RUN: torch-mlir-opt -torch-erase-module-initializer -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK:      module {
// CHECK-NEXT: }
torch.global_slot.module_initializer {
  torch.initialize.global_slots [
  ]
}
