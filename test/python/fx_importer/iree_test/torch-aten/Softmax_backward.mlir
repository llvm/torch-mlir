module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.int, %arg3: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int1 = torch.constant.int 1
    %int6 = torch.constant.int 6
    %0 = torch.aten._softmax_backward_data %arg0, %arg1, %int1, %int6 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
