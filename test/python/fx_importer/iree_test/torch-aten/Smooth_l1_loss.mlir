module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int0 = torch.constant.int 0
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %0 = torch.operator "torch.aten.smooth_l1_loss"(%arg0, %arg1, %int0, %float1.000000e00) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int, !torch.float) -> !torch.vtensor<[128,128],f32> 
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
