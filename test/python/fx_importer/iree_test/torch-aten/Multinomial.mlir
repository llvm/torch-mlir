module {
  func.func @main(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.int) -> !torch.vtensor<[2],si64> {
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.multinomial %arg0, %int2, %false, %none : !torch.vtensor<[4],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[2],si64>
    return %0 : !torch.vtensor<[2],si64>
  }
}
