module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) {
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %none = torch.constant.none
    %0 = torch.aten.normal_functional %arg0, %float0.000000e00, %float1.000000e00, %none : !torch.vtensor<[128,128],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[128,128],f32>
    return %0, %0 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
