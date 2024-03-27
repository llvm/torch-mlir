module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.vtensor<[8],si32>, %arg3: !torch.vtensor<[8,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.operator "torch.aten.index_add"(%arg0, %int0, %arg2, %arg3, %int1) : (!torch.vtensor<[128,128],f32>, !torch.int, !torch.vtensor<[8],si32>, !torch.vtensor<[8,128],f32>, !torch.int) -> !torch.vtensor<[128,128],f32> 
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
