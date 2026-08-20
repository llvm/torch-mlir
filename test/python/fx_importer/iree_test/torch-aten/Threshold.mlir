module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[128,128],f32> {
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %0 = torch.aten.threshold %arg0, %float1.000000e-01, %float1.000000e00 : !torch.vtensor<[128,128],f32>, !torch.float, !torch.float -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
