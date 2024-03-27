module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int) -> !torch.vtensor<[5],f32> {
    %int0 = torch.constant.int 0
    %int30 = torch.constant.int 30
    %int5 = torch.constant.int 5
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %0 = torch.aten.linspace %int0, %int30, %int5, %none, %none_0, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5],f32>
    return %0 : !torch.vtensor<[5],f32>
  }
}
