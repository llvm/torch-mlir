// RUN: npcomp-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  // CHECK: torch.global_slot "private" @float : f64
  torch.attr private "float" : f64
  torch.method private "forward", @method
}

// CHECK: func private @forward() {
func private @method(%arg0: !torch.nn.Module<"c">) {
  return
}

%c42 = std.constant 42.0 : f64
torch.nn_module {
  torch.slot "float", %c42 : f64
} : !torch.nn.Module<"c">
