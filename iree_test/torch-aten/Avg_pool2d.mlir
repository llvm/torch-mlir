module {
  func.func @main(%arg0: !torch.vtensor<[1,1,128,128],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[1,1,64,64],f32> {
    %int2 = torch.constant.int 2
    %int2_0 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int2_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int0_1 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int0, %int0_1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %none = torch.constant.none
    %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[1,1,128,128],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,1,64,64],f32>
    return %3 : !torch.vtensor<[1,1,64,64],f32>
  }
}
