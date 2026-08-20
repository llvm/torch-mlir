module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[],f32> {
    %none = torch.constant.none
    %0 = torch.operator "torch.aten.prod"(%arg0, %none) : (!torch.vtensor<[128,128],f32>, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
}
