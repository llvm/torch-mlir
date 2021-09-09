// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  // CHECK: torch.global_slot "private" @float : !torch.float
  torch.attr private "float" : !torch.float
  torch.method private "forward", @method
}

// CHECK: func private @forward() {
func private @method(%arg0: !torch.nn.Module<"c">) {
  return
}

%c42 = torch.constant.float 42.0
torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"c">
