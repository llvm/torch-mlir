module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],i1>, %arg2: !torch.int) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) {
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %int6 = torch.constant.int 6
    %int0 = torch.constant.int 0
    %cpu = torch.constant.device "cpu"
    %none = torch.constant.none
    %0 = torch.aten.scalar_tensor %float2.000000e00, %int6, %int0, %cpu, %none : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %1 = torch.aten.where.self %arg1, %0, %arg0 : !torch.vtensor<[128,128],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %1, %1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
