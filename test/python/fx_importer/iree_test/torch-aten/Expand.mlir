module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor<[12,128,128],f32> {
    %int12 = torch.constant.int 12
    %int128 = torch.constant.int 128
    %int128_0 = torch.constant.int 128
    %0 = torch.prim.ListConstruct %int12, %int128, %int128_0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.expand %arg0, %0, %false : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[12,128,128],f32>
    return %1 : !torch.vtensor<[12,128,128],f32>
  }
}
