module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128],f32> {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[128,128],f32>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[128],f32>
    return %0 : !torch.vtensor<[128],f32>
  }
}
