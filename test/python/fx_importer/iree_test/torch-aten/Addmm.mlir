module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,256],f32>, %arg2: !torch.vtensor<[256,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %0 = torch.aten.mm %arg1, %arg2 : !torch.vtensor<[128,256],f32>, !torch.vtensor<[256,128],f32> -> !torch.vtensor<[128,128],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.mul.Scalar %0, %int1 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    %int1_0 = torch.constant.int 1
    %2 = torch.aten.mul.Scalar %arg0, %int1_0 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    %int1_1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %1, %2, %int1_1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    return %3 : !torch.vtensor<[128,128],f32>
  }
}
