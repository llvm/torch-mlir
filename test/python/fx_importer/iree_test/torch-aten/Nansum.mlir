module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[],f32> {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[128,128],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
}
