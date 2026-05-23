module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.float) -> !torch.vtensor<[128,128],f32> {
    %int128 = torch.constant.int 128
    %int128_0 = torch.constant.int 128
    %0 = torch.prim.ListConstruct %int128, %int128_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int128_1 = torch.constant.int 128
    %int1 = torch.constant.int 1
    %1 = torch.prim.ListConstruct %int128_1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int6 = torch.constant.int 6
    %int0 = torch.constant.int 0
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %2 = torch.aten.empty_strided %0, %1, %int6, %int0, %cpu, %false : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[128,128],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %3 = torch.aten.fill.Scalar %2, %float1.000000e00 : !torch.vtensor<[128,128],f32>, !torch.float -> !torch.vtensor<[128,128],f32>
    return %3 : !torch.vtensor<[128,128],f32>
  }
}
