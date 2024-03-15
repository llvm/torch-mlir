module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %0 = torch.aten._softmax %arg0, %int1, %false : !torch.vtensor<[128,128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
