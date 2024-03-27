module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> (!torch.vtensor<[128,5],f32>, !torch.vtensor<[128,5],si64>) {
    %int5 = torch.constant.int 5
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %true_0 = torch.constant.bool true
    %values, %indices = torch.aten.topk %arg0, %int5, %int-1, %true, %true_0 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[128,5],f32>, !torch.vtensor<[128,5],si64>
    return %values, %indices : !torch.vtensor<[128,5],f32>, !torch.vtensor<[128,5],si64>
  }
}
