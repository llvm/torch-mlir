module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.float) -> !torch.vtensor<[128,128],f32> {
    %float3.140000e00 = torch.constant.float 3.140000e+00
    %0 = torch.aten.mul.Scalar %arg0, %float3.140000e00 : !torch.vtensor<[128,128],f32>, !torch.float -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
