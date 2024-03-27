module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[],f32> {
    %0 = torch.operator "torch.aten.median"(%arg0) : (!torch.vtensor<[128,128],f32>) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
}
