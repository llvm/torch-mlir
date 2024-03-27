module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %int20 = torch.constant.int 20
    %0 = torch.operator "torch.aten.softplus_backward"(%arg0, %arg1, %int1, %int20) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int, !torch.int) -> !torch.vtensor<[128,128],f32> 
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
