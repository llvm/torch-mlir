// RUN: torch-mlir-opt -torch-globalize-object-graph -split-input-file %s | FileCheck %s

torch.class_type @c {
  // CHECK: torch.global_slot "private" @float : !torch.float
  torch.attr private "float" : !torch.float
  torch.method private "forward", @method
}

// CHECK: func.func private @forward() {
func.func private @method(%arg0: !torch.nn.Module<"c">) {
  return
}

%c42 = torch.constant.float 42.0
torch.nn_module {
  torch.slot "float", %c42 : !torch.float
} : !torch.nn.Module<"c">

func.func private @ensure_all_slots_are_used(%arg0: !torch.nn.Module<"c">) {
  %0 = torch.prim.GetAttr %arg0["float"] : !torch.nn.Module<"c"> -> !torch.float
  return
}
