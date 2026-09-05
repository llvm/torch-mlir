module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,128],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.aten.index_select %arg0, %int0, %arg2 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.vtensor<[4],si64> -> !torch.vtensor<[4,128],f32>
    return %0 : !torch.vtensor<[4,128],f32>
  }
}
