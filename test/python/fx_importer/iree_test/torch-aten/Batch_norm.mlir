module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,1],f32>, %arg2: !torch.vtensor<[128,1],f32>) -> !torch.vtensor<[128,128],f32> {
    %int6 = torch.constant.int 6
    %0 = torch.prims.convert_element_type %arg1, %int6 : !torch.vtensor<[128,1],f32>, !torch.int -> !torch.vtensor<[128,1],f32>
    %int6_0 = torch.constant.int 6
    %1 = torch.prims.convert_element_type %arg2, %int6_0 : !torch.vtensor<[128,1],f32>, !torch.int -> !torch.vtensor<[128,1],f32>
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %int1 = torch.constant.int 1
    %2 = torch.aten.add.Scalar %1, %float1.000000e-05, %int1 : !torch.vtensor<[128,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[128,1],f32>
    %3 = torch.aten.sqrt %2 : !torch.vtensor<[128,1],f32> -> !torch.vtensor<[128,1],f32>
    %4 = torch.aten.reciprocal %3 : !torch.vtensor<[128,1],f32> -> !torch.vtensor<[128,1],f32>
    %int1_1 = torch.constant.int 1
    %5 = torch.aten.mul.Scalar %4, %int1_1 : !torch.vtensor<[128,1],f32>, !torch.int -> !torch.vtensor<[128,1],f32>
    %int1_2 = torch.constant.int 1
    %6 = torch.aten.sub.Tensor %arg0, %0, %int1_2 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,1],f32>, !torch.int -> !torch.vtensor<[128,128],f32>
    %7 = torch.aten.mul.Tensor %6, %5 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,1],f32> -> !torch.vtensor<[128,128],f32>
    return %7 : !torch.vtensor<[128,128],f32>
  }
}
