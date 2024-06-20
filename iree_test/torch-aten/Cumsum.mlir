module {
  func.func @main(%arg0: !torch.vtensor<[1024],f32>, %arg1: !torch.int) -> !torch.vtensor<[1024],f32> {
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %0 = torch.aten.cumsum %arg0, %int0, %none : !torch.vtensor<[1024],f32>, !torch.int, !torch.none -> !torch.vtensor<[1024],f32>
    return %0 : !torch.vtensor<[1024],f32>
  }
}
