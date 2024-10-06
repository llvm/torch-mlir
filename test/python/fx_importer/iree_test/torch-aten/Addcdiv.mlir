module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,1],f32>, %arg2: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.addcdiv %arg0, %arg1, %arg2, %int1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,1],f32>, !torch.vtensor<[1,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
