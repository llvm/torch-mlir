module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],si64> {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %values, %indices = torch.aten.sort %arg0, %int-1, %false : !torch.vtensor<[128,128],f32>, !torch.int, !torch.bool -> !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],si64>
    return %indices : !torch.vtensor<[128,128],si64>
  }
}
