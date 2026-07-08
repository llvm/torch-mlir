module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> {
    %float5.000000e-01 = torch.constant.float 5.000000e-01
    %true = torch.constant.bool true
    %result0, %result1 = torch.aten.native_dropout %arg0, %float5.000000e-01, %true : !torch.vtensor<[128,128],f32>, !torch.float, !torch.bool -> !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],i1>
    return %result0 : !torch.vtensor<[128,128],f32>
  }
}
