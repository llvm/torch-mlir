module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],i1> {
    %0 = torch.aten.isnan %arg0 : !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],i1>
    return %0 : !torch.vtensor<[128,128],i1>
  }
}
