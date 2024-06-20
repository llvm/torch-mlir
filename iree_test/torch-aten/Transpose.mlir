module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
