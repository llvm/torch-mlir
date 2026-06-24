module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[],f32> {
    %int2 = torch.constant.int 2
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.aten.linalg_vector_norm %arg0, %int2, %none, %false, %none_0 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
    return %0 : !torch.vtensor<[],f32>
  }
}
