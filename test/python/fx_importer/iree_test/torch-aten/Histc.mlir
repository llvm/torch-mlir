module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[100],f32> {
    %int100 = torch.constant.int 100
    %int0 = torch.constant.int 0
    %int0_0 = torch.constant.int 0
    %0 = torch.operator "torch.aten.histc"(%arg0, %int100, %int0, %int0_0) : (!torch.vtensor<[128,128],f32>, !torch.int, !torch.int, !torch.int) -> !torch.vtensor<[100],f32> 
    return %0 : !torch.vtensor<[100],f32>
  }
}
