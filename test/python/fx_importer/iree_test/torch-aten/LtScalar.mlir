module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.float) -> !torch.vtensor<[128,128],i1> {
    %float5.000000e-01 = torch.constant.float 5.000000e-01
    %0 = torch.aten.lt.Scalar %arg0, %float5.000000e-01 : !torch.vtensor<[128,128],f32>, !torch.float -> !torch.vtensor<[128,128],i1>
    return %0 : !torch.vtensor<[128,128],i1>
  }
}
