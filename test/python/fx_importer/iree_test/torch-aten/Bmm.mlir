module {
  func.func @main(%arg0: !torch.vtensor<[8,128,8],f32>, %arg1: !torch.vtensor<[8,8,128],f32>) -> !torch.vtensor<[8,128,128],f32> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[8,128,8],f32>, !torch.vtensor<[8,8,128],f32> -> !torch.vtensor<[8,128,128],f32>
    return %0 : !torch.vtensor<[8,128,128],f32>
  }
}