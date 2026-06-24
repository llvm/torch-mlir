module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.aten.tril %arg0, %int0 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
