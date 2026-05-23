module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int5 = torch.constant.int 5
    %0 = torch.aten.pow.Scalar %int5, %arg1 : !torch.int, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
