module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %0 = torch.aten.nan_to_num %arg0, %none, %none_0, %none_1 : !torch.vtensor<[128,128],f32>, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
