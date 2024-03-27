module {
  func.func @main(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[],f32> {
    %int-100 = torch.constant.int -100
    %0 = torch.aten.ne.Scalar %arg1, %int-100 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],i1>
    %int0 = torch.constant.int 0
    %int4 = torch.constant.int 4
    %int0_0 = torch.constant.int 0
    %cpu = torch.constant.device "cpu"
    %none = torch.constant.none
    %1 = torch.aten.scalar_tensor %int0, %int4, %int0_0, %cpu, %none : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],si64>
    %2 = torch.aten.where.self %0, %arg1, %1 : !torch.vtensor<[3],i1>, !torch.vtensor<[3],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[3],si64>
    %int1 = torch.constant.int 1
    %3 = torch.aten.unsqueeze %2, %int1 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3,1],si64>
    %int1_1 = torch.constant.int 1
    %false = torch.constant.bool false
    %4 = torch.aten.gather %arg0, %int1_1, %3, %false : !torch.vtensor<[3,5],f32>, !torch.int, !torch.vtensor<[3,1],si64>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %int1_2 = torch.constant.int 1
    %5 = torch.aten.squeeze.dim %4, %int1_2 : !torch.vtensor<[3,1],f32>, !torch.int -> !torch.vtensor<[3],f32>
    %6 = torch.aten.neg %5 : !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
    %int-100_3 = torch.constant.int -100
    %7 = torch.aten.ne.Scalar %arg1, %int-100_3 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],i1>
    %int0_4 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_5 = torch.constant.int 0
    %cpu_6 = torch.constant.device "cpu"
    %none_7 = torch.constant.none
    %8 = torch.aten.scalar_tensor %int0_4, %int6, %int0_5, %cpu_6, %none_7 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %7, %6, %8 : !torch.vtensor<[3],i1>, !torch.vtensor<[3],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3],f32>
    %int-100_8 = torch.constant.int -100
    %10 = torch.aten.ne.Scalar %arg1, %int-100_8 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],i1>
    %none_9 = torch.constant.none
    %11 = torch.aten.sum %10, %none_9 : !torch.vtensor<[3],i1>, !torch.none -> !torch.vtensor<[],si64>
    %int6_10 = torch.constant.int 6
    %12 = torch.prims.convert_element_type %11, %int6_10 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %none_11 = torch.constant.none
    %13 = torch.aten.sum %9, %none_11 : !torch.vtensor<[3],f32>, !torch.none -> !torch.vtensor<[],f32>
    %14 = torch.aten.div.Tensor %13, %12 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    return %14 : !torch.vtensor<[],f32>
  }
}
