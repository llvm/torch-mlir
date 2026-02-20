module {
  func.func @main(%arg0: !torch.vtensor<[256,128],f32>, %arg1: !torch.int) -> !torch.vtensor<[256,128],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %result0, %result1 = torch.aten.var_mean.correction %arg0, %0, %int0, %true : !torch.vtensor<[256,128],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[256,1],f32>, !torch.vtensor<[256,1],f32>
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %int1_0 = torch.constant.int 1
    %1 = torch.aten.add.Scalar %result0, %float1.000000e-05, %int1_0 : !torch.vtensor<[256,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[256,1],f32>
    %2 = torch.aten.rsqrt %1 : !torch.vtensor<[256,1],f32> -> !torch.vtensor<[256,1],f32>
    %int1_1 = torch.constant.int 1
    %3 = torch.aten.sub.Tensor %arg0, %result1, %int1_1 : !torch.vtensor<[256,128],f32>, !torch.vtensor<[256,1],f32>, !torch.int -> !torch.vtensor<[256,128],f32>
    %4 = torch.aten.mul.Tensor %3, %2 : !torch.vtensor<[256,128],f32>, !torch.vtensor<[256,1],f32> -> !torch.vtensor<[256,128],f32>
    return %4 : !torch.vtensor<[256,128],f32>
  }
}
