module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[128,128],f32> {
    %int3 = torch.constant.int 3
    %0 = torch.aten.pow.Tensor_Scalar %arg0, %int3 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %0 : !torch.vtensor<[128,128],f32>
  }
}
