module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %str = torch.constant.str "none"
    %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[128,128],f32>, !torch.str -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
