module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) {
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %result0, %result1 = torch.aten.var_mean.correction %arg0, %none, %int1, %false : !torch.vtensor<[128,128],f32>, !torch.none, !torch.int, !torch.bool -> !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
    return %result0, %result1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
  }
}
