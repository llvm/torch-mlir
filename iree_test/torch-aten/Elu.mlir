module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %int1 = torch.constant.int 1
    %int1_0 = torch.constant.int 1
    %0 = torch.aten.elu %arg0, %float1.000000e00, %int1, %int1_0 : !torch.vtensor<[128,128],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
