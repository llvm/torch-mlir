module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[128],f32>
    return %1 : !torch.vtensor<[128],f32>
  }
}
