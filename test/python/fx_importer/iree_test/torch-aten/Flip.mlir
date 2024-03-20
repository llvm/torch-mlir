module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.flip %arg0, %0 : !torch.vtensor<[128,128],f32>, !torch.list<int> -> !torch.vtensor<[128,128],f32>
    return %1 : !torch.vtensor<[128,128],f32>
  }
}