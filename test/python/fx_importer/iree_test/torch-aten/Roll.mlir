module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int1_0 = torch.constant.int 1
    %1 = torch.prim.ListConstruct %int0, %int1_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.aten.roll %arg0, %0, %1 : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.list<int> -> !torch.vtensor<[128,128],f32>
    return %2 : !torch.vtensor<[128,128],f32>
  }
}
