module {
  func.func @main(%arg0: !torch.vtensor<[128,128],i1>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.aten.where.self %arg0, %arg1, %arg2 : !torch.vtensor<[128,128],i1>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
