module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.float) -> !torch.vtensor<[128,128],f32> {
    %float3.140000e00 = torch.constant.float 3.140000e+00
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %false = torch.constant.bool false
    %none_2 = torch.constant.none
    %0 = torch.aten.full_like %arg0, %float3.140000e00, %none, %none_0, %none_1, %false, %none_2 : !torch.vtensor<[128,128],f32>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
