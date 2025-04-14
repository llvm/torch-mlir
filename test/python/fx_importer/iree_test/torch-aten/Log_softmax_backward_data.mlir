module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.aten.exp %arg1 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %int0 = torch.constant.int 0
    %1 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %2 = torch.aten.sum.dim_IntList %arg0, %1, %true, %none : !torch.vtensor<[128,128],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,128],f32>
    %3 = torch.aten.mul.Tensor %0, %2 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[1,128],f32> -> !torch.vtensor<[128,128],f32>
    %int1 = torch.constant.int 1
    %4 = torch.aten.sub.Tensor %arg0, %3, %int1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %4 : !torch.vtensor<[128,128],f32>
  }
}
