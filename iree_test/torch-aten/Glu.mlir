module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,64],f32> {
    %int-1 = torch.constant.int -1
    %0 = torch.aten.glu %arg0, %int-1 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,64],f32>
    return %0 : !torch.vtensor<[128,64],f32>
  }
}
