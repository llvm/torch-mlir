module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %none = torch.constant.none
    %0 = torch.operator "torch.aten.logit_backward"(%arg0, %arg1, %none) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.none) -> !torch.vtensor<[128,128],f32> 
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
