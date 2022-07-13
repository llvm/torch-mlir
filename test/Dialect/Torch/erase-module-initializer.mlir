// RUN: torch-mlir-opt -torch-erase-module-initializer -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK:      module {
// CHECK-NEXT: }
torch.global_slot.module_initializer {
  torch.initialize.global_slots [
  ]
}

// -----

torch.global_slot @slot0 : !torch.int

// expected-error@+1 {{could not erase non-empty module initializer}}
torch.global_slot.module_initializer {
  %0 = torch.constant.int 0
  torch.initialize.global_slots [
    @slot0(%0: !torch.int)
  ]
}
