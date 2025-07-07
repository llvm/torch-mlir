module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.aten.log %arg0 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
