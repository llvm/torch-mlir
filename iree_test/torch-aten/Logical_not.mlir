module {
  func.func @main(%arg0: !torch.vtensor<[8,128],f32>) -> !torch.vtensor<[8,128],i1> {
    %0 = torch.aten.logical_not %arg0 : !torch.vtensor<[8,128],f32> -> !torch.vtensor<[8,128],i1>
    return %0 : !torch.vtensor<[8,128],i1>
  }
}