module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[2,8256],si64> {
    %int128 = torch.constant.int 128
    %int128_0 = torch.constant.int 128
    %int0 = torch.constant.int 0
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %0 = torch.operator "torch.aten.tril_indices"(%int128, %int128_0, %int0, %int4, %none, %cpu, %false) : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.none, !torch.Device, !torch.bool) -> !torch.vtensor<[2,8256],si64> 
    return %0 : !torch.vtensor<[2,8256],si64>
  }
}
