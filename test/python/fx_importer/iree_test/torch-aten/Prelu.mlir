module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %int128 = torch.constant.int 128
    %0 = torch.prim.ListConstruct %int1, %int128 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %arg1, %0 : !torch.vtensor<[128],f32>, !torch.list<int> -> !torch.vtensor<[1,128],f32>
    %int0 = torch.constant.int 0
    %2 = torch.aten.gt.Scalar %arg0, %int0 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],i1>
    %3 = torch.aten.mul.Tensor %1, %arg0 : !torch.vtensor<[1,128],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    %4 = torch.aten.where.self %2, %arg0, %3 : !torch.vtensor<[128,128],i1>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %4 : !torch.vtensor<[128,128],f32>
  }
}
