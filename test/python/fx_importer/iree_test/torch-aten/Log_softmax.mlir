module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amax %arg0, %0, %true : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,128],f32>
    %int1 = torch.constant.int 1
    %2 = torch.aten.sub.Tensor %arg0, %1, %int1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[1,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    %3 = torch.aten.exp %2 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %int0_0 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0_0 : (!torch.int) -> !torch.list<int>
    %true_1 = torch.constant.bool true
    %none = torch.constant.none
    %5 = torch.aten.sum.dim_IntList %3, %4, %true_1, %none : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,128],f32>
    %6 = torch.aten.log %5 : !torch.vtensor<[1,128],f32> -> !torch.vtensor<[1,128],f32>
    %int1_2 = torch.constant.int 1
    %7 = torch.aten.sub.Tensor %2, %6, %int1_2 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[1,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %7 : !torch.vtensor<[128,128],f32>
  }
}
