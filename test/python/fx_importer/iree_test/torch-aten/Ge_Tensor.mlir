module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],i1> {
    %0 = torch.aten.ge.Tensor %arg0, %arg1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],i1>
    return %0 : !torch.vtensor<[128,128],i1>
  }
}
