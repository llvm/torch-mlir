module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.float) -> !torch.vtensor<[128,128],f32> {
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %0 = torch.aten.threshold_backward %arg0, %arg1, %float1.000000e-01 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.float -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}