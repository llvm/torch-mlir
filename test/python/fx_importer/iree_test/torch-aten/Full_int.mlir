module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[128,128],si64> {
    %int128 = torch.constant.int 128
    %int128_0 = torch.constant.int 128
    %0 = torch.prim.ListConstruct %int128, %int128_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int128_1 = torch.constant.int 128
    %int1 = torch.constant.int 1
    %1 = torch.prim.ListConstruct %int128_1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %int0 = torch.constant.int 0
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %2 = torch.aten.empty_strided %0, %1, %int4, %int0, %cpu, %false : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[128,128],si64>
    %int2 = torch.constant.int 2
    %3 = torch.aten.fill.Scalar %2, %int2 : !torch.vtensor<[128,128],si64>, !torch.int -> !torch.vtensor<[128,128],si64>
    return %3 : !torch.vtensor<[128,128],si64>
  }
}
