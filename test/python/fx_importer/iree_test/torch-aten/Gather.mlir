module {
  func.func @main(%arg0: !torch.vtensor<[3,2],f32>, %arg1: !torch.int, %arg2: !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[3,2],f32> {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %0 = torch.aten.gather %arg0, %int1, %arg2, %false : !torch.vtensor<[3,2],f32>, !torch.int, !torch.vtensor<[3,2],si64>, !torch.bool -> !torch.vtensor<[3,2],f32>
    return %0 : !torch.vtensor<[3,2],f32>
  }
}
