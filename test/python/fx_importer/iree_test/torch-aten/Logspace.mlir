module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[1],f32> {
    %int0 = torch.constant.int 0
    %int128 = torch.constant.int 128
    %int1 = torch.constant.int 1
    %float1.000000e01 = torch.constant.float 1.000000e+01
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %0 = torch.operator "torch.aten.logspace"(%int0, %int128, %int1, %float1.000000e01, %none, %none_0, %cpu, %false) : (!torch.int, !torch.int, !torch.int, !torch.float, !torch.none, !torch.none, !torch.Device, !torch.bool) -> !torch.vtensor<[1],f32> 
    return %0 : !torch.vtensor<[1],f32>
  }
}
