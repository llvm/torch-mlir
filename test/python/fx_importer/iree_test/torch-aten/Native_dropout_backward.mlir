module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %float3.000000e00 = torch.constant.float 3.000000e+00
    %0 = torch.aten.native_dropout_backward %arg0, %arg1, %float3.000000e00 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.float -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
