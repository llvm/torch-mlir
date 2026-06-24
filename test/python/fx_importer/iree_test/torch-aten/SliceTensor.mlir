module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int) -> !torch.vtensor<[128,2],f32> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int8 = torch.constant.int 8
    %int4 = torch.constant.int 4
    %0 = torch.aten.slice.Tensor %arg0, %int1, %int0, %int8, %int4 : !torch.vtensor<[128,128],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[128,2],f32>
    return %0 : !torch.vtensor<[128,2],f32>
  }
}
