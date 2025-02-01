module {
  func.func @main(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.bool) -> !torch.vtensor<[3200,128],f32> {
    %int4 = torch.constant.int 4
    %0 = torch.prims.convert_element_type %arg1, %int4 : !torch.vtensor<[128,128],f32>, !torch.int -> !torch.vtensor<[128,128],si64>
    %int3200 = torch.constant.int 3200
    %1 = torch.prim.ListConstruct %int3200 : (!torch.int) -> !torch.list<int>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %false = torch.constant.bool false
    %2 = torch.aten.new_zeros %0, %1, %none, %none_0, %none_1, %false : !torch.vtensor<[128,128],si64>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[3200],si64>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %false_5 = torch.constant.bool false
    %none_6 = torch.constant.none
    %3 = torch.aten.ones_like %0, %none_2, %none_3, %none_4, %false_5, %none_6 : !torch.vtensor<[128,128],si64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[128,128],si64>
    %4 = torch.prim.ListConstruct %0 : (!torch.vtensor<[128,128],si64>) -> !torch.list<optional<vtensor>>
    %true = torch.constant.bool true
    %5 = torch.operator "torch.aten._unsafe_index_put"(%2, %4, %3, %true) : (!torch.vtensor<[3200],si64>, !torch.list<optional<vtensor>>, !torch.vtensor<[128,128],si64>, !torch.bool) -> !torch.vtensor<[3200],si64> 
    %6 = torch.prim.ListConstruct %0 : (!torch.vtensor<[128,128],si64>) -> !torch.list<optional<vtensor>>
    %7 = torch.aten.index.Tensor %5, %6 : !torch.vtensor<[3200],si64>, !torch.list<optional<vtensor>> -> !torch.vtensor<[128,128],si64>
    %int-1 = torch.constant.int -1
    %8 = torch.aten.unsqueeze %7, %int-1 : !torch.vtensor<[128,128],si64>, !torch.int -> !torch.vtensor<[128,128,1],si64>
    %9 = torch.aten.div.Tensor %arg0, %8 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128,1],si64> -> !torch.vtensor<[128,128,128],f32>
    %int-1_7 = torch.constant.int -1
    %10 = torch.aten.eq.Scalar %0, %int-1_7 : !torch.vtensor<[128,128],si64>, !torch.int -> !torch.vtensor<[128,128],i1>
    %int-1_8 = torch.constant.int -1
    %11 = torch.aten.unsqueeze %10, %int-1_8 : !torch.vtensor<[128,128],i1>, !torch.int -> !torch.vtensor<[128,128,1],i1>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %int6 = torch.constant.int 6
    %int0 = torch.constant.int 0
    %cpu = torch.constant.device "cpu"
    %none_9 = torch.constant.none
    %12 = torch.aten.scalar_tensor %float0.000000e00, %int6, %int0, %cpu, %none_9 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %13 = torch.aten.where.self %11, %12, %9 : !torch.vtensor<[128,128,1],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[128,128,128],f32> -> !torch.vtensor<[128,128,128],f32>
    %int3200_10 = torch.constant.int 3200
    %int128 = torch.constant.int 128
    %14 = torch.prim.ListConstruct %int3200_10, %int128 : (!torch.int, !torch.int) -> !torch.list<int>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %false_14 = torch.constant.bool false
    %15 = torch.aten.new_zeros %9, %14, %none_11, %none_12, %none_13, %false_14 : !torch.vtensor<[128,128,128],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[3200,128],f32>
    %16 = torch.prim.ListConstruct %0 : (!torch.vtensor<[128,128],si64>) -> !torch.list<optional<vtensor>>
    %true_15 = torch.constant.bool true
    %17 = torch.operator "torch.aten._unsafe_index_put"(%15, %16, %13, %true_15) : (!torch.vtensor<[3200,128],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[128,128,128],f32>, !torch.bool) -> !torch.vtensor<[3200,128],f32> 
    return %17 : !torch.vtensor<[3200,128],f32>
  }
}
