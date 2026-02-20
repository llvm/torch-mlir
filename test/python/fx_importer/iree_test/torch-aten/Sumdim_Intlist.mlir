module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[128],f32>
    return %1 : !torch.vtensor<[128],f32>
  }
}
