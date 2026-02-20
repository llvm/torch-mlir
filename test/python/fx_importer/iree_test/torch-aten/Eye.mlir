module {
  func.func @main(%arg0: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int128 = torch.constant.int 128
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %0 = torch.aten.eye %int128, %none, %none_0, %cpu, %false : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
