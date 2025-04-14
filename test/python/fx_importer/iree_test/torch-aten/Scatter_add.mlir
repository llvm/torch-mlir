module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.vtensor<[128,128],si64>, %arg3: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.scatter_add %arg0, %int1, %arg2, %arg3 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.vtensor<[128,128],si64>, !torch.vtensor<[128,128],f32> -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
