module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.float) -> !torch.vtensor<[128,128],i1> {
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %0 = torch.aten.le.Scalar %arg0, %float1.000000e00 : !torch.vtensor<[128,128],f32>, !torch.float -> !torch.vtensor<[128,128],i1>
    return %0 : !torch.vtensor<[128,128],i1>
  }
}
