module {
  func.func @main(%arg0: !torch.vtensor<[3],si8>, %arg1: !torch.vtensor<[3],si8>) -> !torch.vtensor<[3],si8> {
    %0 = torch.aten.bitwise_xor.Tensor %arg0, %arg1 : !torch.vtensor<[3],si8>, !torch.vtensor<[3],si8> -> !torch.vtensor<[3],si8>
    return %0 : !torch.vtensor<[3],si8>
  }
}
