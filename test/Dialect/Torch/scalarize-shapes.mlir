// RUN: torch-mlir-opt <%s --torch-scalarize-shapes -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @shape_as_tensor
func.func @shape_as_tensor(%arg0 : !torch.vtensor<[5,?,?],f32>) -> !torch.vtensor<[3],si32> {
    // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
    // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
    // CHECK-DAG: %[[I2:.+]] = torch.constant.int 2
    // CHECK-DAG: %[[I5:.+]] = torch.constant.int 5
    // CHECK-DAG: %[[I1:.+]] = torch.constant.int 1
    // CHECK-DAG: %[[SZ1:.+]] = torch.aten.size.int %arg0, %[[I1]]
    // CHECK-DAG: %[[SZ2:.+]] = torch.aten.size.int %arg0, %[[I2]]
    // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[I5]], %[[SZ1]], %[[SZ2]]
    // CHECK-DAG: %[[TENSOR:.+]] = torch.aten.tensor %[[LIST]], %[[NONE]], %[[NONE]], %[[FALSE]]
    // CHECK: return %[[TENSOR]] : !torch.vtensor<[3],si32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %literal1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %0 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?],f32> -> !torch.vtensor<[3],si32>
    %1 = torch.aten.index_select %0, %int0, %literal1: !torch.vtensor<[3],si32>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si32>
    %2 = torch.aten.item %1 : !torch.vtensor<[1],si32> -> !torch.int
    %3 = torch.prim.ListConstruct %2 : (!torch.int) -> !torch.list<int>
    return %0 : !torch.vtensor<[3],si32>
}

// -----

// CHECK-LABEL: @shape_as_tensor_dim
func.func @shape_as_tensor_dim(%arg0 : !torch.vtensor<[5,?,?],f32>) -> !torch.vtensor<[],si32> {
    // CHECK: %[[INT1:.+]] = torch.constant.int 1
    // CHECK: %[[SZ:.+]] = torch.aten.size.int %arg0, %[[INT1]]
    // CHECK: %[[INT1_0:.+]] = torch.constant.int 1
    // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
    // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
    // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT1_0]]
    // CHECK: %[[TENSOR:.+]] = torch.aten.full %[[LIST]], %[[SZ]], %[[NONE]], %[[NONE]], %[[NONE]], %[[FALSE]]
    // CHECK: return %[[TENSOR]] : !torch.vtensor<[],si32>
    %shape = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?],f32> -> !torch.vtensor<[3],si32>
    %dim = torch.constant.int 0
    %idx = torch.vtensor.literal(dense<1> : tensor<si32>) : !torch.vtensor<[],si32>
    %select = torch.aten.index_select %shape, %dim, %idx : !torch.vtensor<[3],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[],si32>
    %item = torch.aten.item %select : !torch.vtensor<[],si32> -> !torch.int
    %list = torch.prim.ListConstruct %item : (!torch.int) -> !torch.list<int>
    return %select : !torch.vtensor<[],si32>
}


// -----

// CHECK-LABEL: @shape_as_tensor_dim_item
func.func @shape_as_tensor_dim_item(%arg0 : !torch.vtensor<[5,?,?],f32>) -> !torch.int {
    // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
    // CHECK-DAG: %[[SZ:.+]] = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[5,?,?],f32>, !torch.int -> !torch.int
    // CHECK: return %[[SZ]]
    %shape = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?],f32> -> !torch.vtensor<[3],si32>
    %dim = torch.constant.int 0
    %idx = torch.vtensor.literal(dense<1> : tensor<si32>) : !torch.vtensor<[],si32>
    %select = torch.aten.index_select %shape, %dim, %idx : !torch.vtensor<[3],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[],si32>
    %out = torch.aten.item %select : !torch.vtensor<[],si32> -> !torch.int
    %list = torch.prim.ListConstruct %out : (!torch.int) -> !torch.list<int>
    return %out : !torch.int
}

// -----

// CHECK-LABEL: @literal_item
func.func @literal_item() -> !torch.int {
    // CHECK: %int2 = torch.constant.int 2
    // CHECK: return %int2 : !torch.int
    %shape = torch.vtensor.literal(dense<[1,2,3]> : tensor<3xsi32>) : !torch.vtensor<[3],si32>
    %dim = torch.constant.int 0
    %idx = torch.vtensor.literal(dense<1> : tensor<si32>) : !torch.vtensor<[],si32>
    %select = torch.aten.index_select %shape, %dim, %idx : !torch.vtensor<[3],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[],si32>
    %out = torch.aten.item %select : !torch.vtensor<[],si32> -> !torch.int
    %list = torch.prim.ListConstruct %out : (!torch.int) -> !torch.list<int>
    return %out : !torch.int
}

// -----

// CHECK-LABEL: @arith_prop
func.func @arith_prop(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    // CHECK: %[[float0:.*]] = torch.constant.float 0.000000e+00
    // CHECK: %[[int0:.*]] = torch.constant.int 0
    // CHECK: %[[x0:.*]] = torch.aten.size.int %arg0, %[[int0]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[int1:.*]] = torch.constant.int 1
    // CHECK: %[[x1:.*]] = torch.aten.size.int %arg0, %[[int1]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[int12:.*]] = torch.constant.int 12
    // CHECK: %[[int1_0:.*]] = torch.constant.int 1
    // CHECK: %[[x2:.*]] = torch.aten.floordiv.int %[[x0]], %[[int12]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[x3:.*]] = torch.aten.floordiv.int %[[x1]], %[[int1_0]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[int12_1:.*]] = torch.constant.int 12
    // CHECK: %[[int1_2:.*]] = torch.constant.int 1
    // CHECK: %[[x4:.*]] = torch.aten.mul.int %[[x2]], %[[int12_1]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[x5:.*]] = torch.aten.mul.int %[[x3]], %[[int1_2]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[x6:.*]] = torch.aten.sub.int %[[x0]], %[[x4]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[x7:.*]] = torch.aten.sub.int %[[x1]], %[[x5]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[x8:.*]] = torch.prim.ListConstruct %[[x7]], %[[x6]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[x9:.*]] = torch.aten.constant_pad_nd %arg0, %[[x8]], %[[float0]] : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?,?],f32>
    // CHECK: return %[[x9]] : !torch.vtensor<[?,?],f32>
    %0 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %int1 = torch.constant.int 1
    %2 = torch.vtensor.literal(dense<[12, 1]> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
    %int0 = torch.constant.int 0
    %3 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[2],si64>
    %4 = torch.aten.div.Tensor %3, %2 : !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64> -> !torch.vtensor<[2],si64>
    %5 = torch.aten.mul.Tensor %4, %2 : !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64> -> !torch.vtensor<[2],si64>
    %6 = torch.aten.sub.Tensor %3, %5, %int1 : !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[2],si64>
    %7 = torch.aten.index_select %6, %int0, %1 : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %8 = torch.aten.index_select %6, %int0, %0 : !torch.vtensor<[2],si64>, !torch.int, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %9 = torch.aten.item %7 : !torch.vtensor<[],si64> -> !torch.int
    %10 = torch.aten.item %8 : !torch.vtensor<[],si64> -> !torch.int
    %11 = torch.prim.ListConstruct %10, %9 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.aten.constant_pad_nd %arg0, %11, %float0.000000e00 : !torch.vtensor<[?,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[?,?],f32>
    return %12 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: @broadcast_prop
func.func @broadcast_prop(%arg0 : !torch.vtensor<[?,?],f32>) -> !torch.int {
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[SZE:.*]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    // CHECK: return %[[SZE]] : !torch.int
    %dim = torch.constant.int 0
    %size = torch.aten.size.int %arg0, %dim : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    %shape = torch.prim.NumToTensor.Scalar %size : !torch.int -> !torch.vtensor<[],si32>
    %int3 = torch.constant.int 3
    %idx = torch.vtensor.literal(dense<-1> : tensor<si32>) : !torch.vtensor<[],si32>
    %bcastlist = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
    %bcast = torch.aten.broadcast_to %shape, %bcastlist : !torch.vtensor<[],si32>, !torch.list<int> -> !torch.vtensor<[3],si32>
    %select = torch.aten.index_select %bcast, %dim, %idx : !torch.vtensor<[3],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[],si32>
    %out = torch.aten.item %select : !torch.vtensor<[],si32> -> !torch.int
    %list = torch.prim.ListConstruct %out : (!torch.int) -> !torch.list<int>
    return %out : !torch.int
}

// -----

// CHECK-LABEL: @eq_int_fold
func.func @eq_int_fold(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,1],f32> {
    // CHECK: %[[int1:.*]] = torch.constant.int 1
    // CHECK: %[[int0:.*]] = torch.constant.int 0
    // CHECK: %[[sze0:.*]] = torch.aten.size.int %arg0, %[[int0]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[sze1:.*]] = torch.aten.size.int %arg0, %[[int1]] : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[mul:.*]] = torch.aten.mul.int %[[sze0]], %[[sze1]] : !torch.int, !torch.int -> !torch.int
    // CHECK: %[[gt0:.*]] = torch.aten.gt.int %[[sze0]], %[[int0]] : !torch.int, !torch.int -> !torch.bool
    // CHECK: torch.runtime.assert %[[gt0]], "Expected dim size > 0."
    // CHECK: %[[gt1:.*]] = torch.aten.gt.int %[[sze1]], %[[int0]] : !torch.int, !torch.int -> !torch.bool
    // CHECK: torch.runtime.assert %[[gt1]], "Expected dim size > 0."
    // CHECK: %[[list:.*]] = torch.prim.ListConstruct %[[mul]], %[[int1]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[view:.*]] = torch.aten.view %arg0, %[[list]] : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
    // CHECK: return %[[view:.*]] : !torch.vtensor<[?,1],f32>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
    %2 = torch.aten.mul.int %0, %1 : !torch.int, !torch.int -> !torch.int
    %3 = torch.aten.eq.int %2, %int0 : !torch.int, !torch.int -> !torch.bool
    %4 = torch.aten.Int.bool %3 : !torch.bool -> !torch.int
    %5 = torch.prim.NumToTensor.Scalar %4 : !torch.int -> !torch.vtensor<[],i1>
    %6 = torch.prim.NumToTensor.Scalar %0 : !torch.int -> !torch.vtensor<[],si64>
    %7 = torch.prim.NumToTensor.Scalar %2 : !torch.int -> !torch.vtensor<[],si64>
    %8 = torch.aten.where.self %5, %6, %7 : !torch.vtensor<[],i1>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %9 = torch.aten.item %8 : !torch.vtensor<[],si64> -> !torch.int
    %10 = torch.prim.ListConstruct %9, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %11 = torch.aten.view %arg0, %10 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
    return %11 : !torch.vtensor<[?,1],f32>
}

// -----

// CHECK-LABEL: @shape_as_tensor_slice
func.func @shape_as_tensor_slice(%arg0 : !torch.vtensor<[5,?,?,?],f32>) -> !torch.vtensor<[2],si32> {
    // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
    // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
    // CHECK-DAG: %[[INT3:.+]] = torch.constant.int 3
    // CHECK-DAG: %[[INT1:.+]] = torch.constant.int 1
    // CHECK-DAG: %[[SZ1:.+]] = torch.aten.size.int %arg0, %[[INT1]]
    // CHECK-DAG: %[[SZ3:.+]] = torch.aten.size.int %arg0, %[[INT3]]
    // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[SZ1]], %[[SZ3]]
    // CHECK-DAG: %[[TENSOR:.+]] = torch.aten.tensor %[[LIST]], %[[NONE]], %[[NONE]], %[[FALSE]]
    // CHECK: return %[[TENSOR]]
    %idx = torch.vtensor.literal(dense<1> : tensor<si32>) : !torch.vtensor<[],si32>
    %shape = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?,?],f32> -> !torch.vtensor<[4],si32>
    %dim = torch.constant.int 0
    %start = torch.constant.int 1
    %end = torch.constant.int 5
    %step = torch.constant.int 2
    %slice = torch.aten.slice.Tensor %shape, %dim, %start, %end, %step : !torch.vtensor<[4], si32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2], si32>
    %select = torch.aten.index_select %slice, %dim, %idx : !torch.vtensor<[2],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[],si32>
    %item = torch.aten.item %select : !torch.vtensor<[],si32> -> !torch.int
    %list = torch.prim.ListConstruct %item : (!torch.int) -> !torch.list<int>
    return %slice : !torch.vtensor<[2],si32>
}


// -----

// CHECK-LABEL: @view_as_flatten_static
func.func @view_as_flatten_static(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[?,?,1024],f32> {
    // CHECK-DAG:   %[[TWO:.*]] = torch.constant.int 2
    // CHECK-DAG:   %[[THREE:.*]] = torch.constant.int 3
    // CHECK-DAG:   %[[FLAT:.*]] = torch.aten.flatten.using_ints %arg0, %[[TWO]], %[[THREE]] : !torch.vtensor<[?,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,1024],f32>
    // CHECK:   return %[[FLAT]] : !torch.vtensor<[?,?,1024],f32>
    %int1024 = torch.constant.int 1024
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    %2 = torch.prim.ListConstruct %0, %1, %int1024 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?,16,64],f32>, !torch.list<int> -> !torch.vtensor<[?,?,1024],f32>
    return %3 : !torch.vtensor<[?,?,1024],f32>
}


// -----

// CHECK-LABEL: @view_as_unflatten_static
func.func @view_as_unflatten_static(%arg0: !torch.vtensor<[?,?,1024],f32>) -> !torch.vtensor<[?,?,16,64],f32> {
    // CHECK-DAG:   %[[TWO:.*]] = torch.constant.int 2
    // CHECK-DAG:   %[[CST16:.*]] = torch.constant.int 16
    // CHECK-DAG:   %[[CST64:.*]] = torch.constant.int 64
    // CHECK:   %[[LIST:.*]] = torch.prim.ListConstruct %[[CST16]], %[[CST64]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK:   %[[FLAT:.*]] = torch.aten.unflatten.int %arg0, %[[TWO]], %[[LIST]] : !torch.vtensor<[?,?,1024],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[?,?,16,64],f32>
    // CHECK:   return %[[FLAT]] : !torch.vtensor<[?,?,16,64],f32>
    %int16 = torch.constant.int 16
    %int64 = torch.constant.int 64
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,1024],f32>, !torch.int -> !torch.int
    %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,1024],f32>, !torch.int -> !torch.int
    %2 = torch.prim.ListConstruct %0, %1, %int16, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?,1024],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16,64],f32>
    return %3 : !torch.vtensor<[?,?,16,64],f32>
}


// -----

// CHECK-LABEL: @view_as_flatten_dynamic
func.func @view_as_flatten_dynamic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
    // CHECK-DAG:   %[[TWO:.*]] = torch.constant.int 2
    // CHECK-DAG:   %[[THREE:.*]] = torch.constant.int 3
    // CHECK-DAG:   %[[FLAT:.*]] = torch.aten.flatten.using_ints %arg0, %[[TWO]], %[[THREE]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
    // CHECK:   return %[[FLAT]] : !torch.vtensor<[?,?,?],f32>
    %int-1 = torch.constant.int -1
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
    %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
    %2 = torch.prim.ListConstruct %0, %1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>
    return %3 : !torch.vtensor<[?,?,?],f32>
}


// -----

// CHECK-LABEL: @unsqueeze_squeeze_combo
func.func @unsqueeze_squeeze_combo(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.int {
    // CHECK: %int0 = torch.constant.int 0
    // CHECK: %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    // CHECK: return %0 : !torch.int
    %0 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %1 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<1024> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %3 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,?,16,64],f32> -> !torch.vtensor<[4],si64>
    %4 = torch.aten.index_select %3, %int0, %1 : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %5 = torch.aten.squeeze.dim %4, %int0 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[],si64>
    %6 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,?,16,64],f32> -> !torch.vtensor<[4],si64>
    %7 = torch.aten.index_select %6, %int0, %0 : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %8 = torch.aten.squeeze.dim %7, %int0 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[],si64>
    %9 = torch.aten.unsqueeze %5, %int0 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %10 = torch.aten.unsqueeze %8, %int0 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %11 = torch.prim.ListConstruct %9, %10, %2 : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.list<vtensor>
    %12 = torch.aten.cat %11, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[3],si64>
    %13 = torch.aten.slice.Tensor %12, %int0, %int0, %int1, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %14 = torch.aten.item %13 : !torch.vtensor<[1],si64> -> !torch.int
    %list = torch.prim.ListConstruct %14 : (!torch.int) -> !torch.list<int>
    return %14 : !torch.int
}


// -----

// CHECK-LABEL: @eq_tensor_and_where_self
func.func @eq_tensor_and_where_self(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[4],si64> {
    // CHECK: %[[I1:.*]] = torch.constant.int 1
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[DIM1:.*]] = torch.aten.size.int %arg0, %[[I1]] : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    // CHECK: %[[DIM0:.*]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    // CHECK: %[[I1_0:.*]] = torch.constant.int 1
    // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[I1_0]], %[[DIM1]], %[[DIM1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[none:.*]] = torch.constant.none
    // CHECK: %[[false:.*]] = torch.constant.bool false
    // CHECK: %[[TENSOR:.*]] = torch.aten.tensor %[[LIST]], %[[none]], %[[none]], %[[false]] : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[4],si64>
    // CHECK: return %[[TENSOR]] : !torch.vtensor<[4],si64>
    %none = torch.constant.none
    %0 = torch.vtensor.literal(dense<-1> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %idx = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %2 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    %3 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    %4 = torch.prim.ListConstruct %3, %int1, %2, %2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.tensor %4, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[4],si64>
    %6 = torch.aten.eq.Tensor %5, %0 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
    %7 = torch.aten.where.self %6, %1, %5 : !torch.vtensor<[4],i1>, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
    %select = torch.aten.index_select %7, %int0, %idx : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %item = torch.aten.item %select : !torch.vtensor<[],si64> -> !torch.int
    %list = torch.prim.ListConstruct %item : (!torch.int) -> !torch.list<int>
    return %7 : !torch.vtensor<[4],si64>
}


// -----

// CHECK-LABEL: @eq_tensor_from_tensor_and_literal
func.func @eq_tensor_from_tensor_and_literal(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[4],i1> {
    // CHECK: %[[int1:.*]] = torch.constant.int 1
    // CHECK: %[[int0:.*]] = torch.constant.int 0
    // CHECK: %[[int1_0:.*]] = torch.constant.int 1
    // CHECK: %[[int0_1:.*]] = torch.constant.int 0
    // CHECK: %[[int0_2:.*]] = torch.constant.int 0
    // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[int0]], %[[int1_0]], %[[int0_1]], %[[int0_2]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[none:.*]] = torch.constant.none
    // CHECK: %[[false:.*]] = torch.constant.bool false
    // CHECK: %[[TENSOR:.*]] = torch.aten.tensor %[[LIST]], %[[none]], %[[none]], %[[false]] : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[4],i1>
    // CHECK: return %[[TENSOR]] : !torch.vtensor<[4],i1>
    %none = torch.constant.none
    %0 = torch.vtensor.literal(dense<-1> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %idx = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %int-1 = torch.constant.int -1
    %int0 = torch.constant.int 0
    %2 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    %3 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    %4 = torch.prim.ListConstruct %3, %int-1, %2, %2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.tensor %4, %none, %none, %false : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[4],si64>
    %6 = torch.aten.eq.Tensor %5, %0 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],i1>
    %select = torch.aten.index_select %6, %int0, %idx : !torch.vtensor<[4],i1>, !torch.int, !torch.vtensor<[],si64> -> !torch.vtensor<[],i1>
    %item = torch.aten.item %select : !torch.vtensor<[],i1> -> !torch.int
    %list = torch.prim.ListConstruct %item : (!torch.int) -> !torch.list<int>
    return %6 : !torch.vtensor<[4],i1>
}



// -----

// CHECK-LABEL: @squeeze_dim_full_fold
func.func @squeeze_dim_full_fold(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.list<int> {
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[SZE:.*]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    // CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[SZE]] : (!torch.int) -> !torch.list<int>
    // CHECK: return %[[LIST]] : !torch.list<int>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %false = torch.constant.bool false
    %51 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?],si64>, !torch.int -> !torch.int
    %55 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %56 = torch.aten.full %55, %51, %none, %none, %none, %false : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[1],si64>
    %57 = torch.aten.squeeze.dim %56, %int0 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[],si64>
    %58 = torch.aten.item %57 : !torch.vtensor<[],si64> -> !torch.int
    %59 = torch.prim.ListConstruct %58 : (!torch.int) -> !torch.list<int>
    return %59 : !torch.list<int>
}

// -----

// CHECK-LABEL: @pytorch_dynamic_pad_export_view$prop(
func.func @pytorch_dynamic_pad_export_view$prop(%arg0: !torch.vtensor<[?,144,?,?],f32>) -> !torch.vtensor<[4,2],si64> {
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[x0:.*]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I2:.*]] = torch.constant.int 2
    // CHECK: %[[x1:.*]] = torch.aten.size.int %arg0, %[[I2]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I3:.*]] = torch.constant.int 3
    // CHECK: %[[x2:.*]] = torch.aten.size.int %arg0, %[[I3]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I144:.*]] = torch.constant.int 144
    // CHECK: %[[I0_0:.*]] = torch.constant.int 0
    // CHECK: %[[I0_1:.*]] = torch.constant.int 0
    // CHECK: %[[I0_2:.*]] = torch.constant.int 0
    // CHECK: %[[I0_3:.*]] = torch.constant.int 0
    // CHECK: %[[x3:.*]] = torch.prim.ListConstruct %[[x0]], %[[I144]], %[[x1]], %[[x2]], %[[I0_0]], %[[I0_1]], %[[I0_2]], %[[I0_3]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[none:.*]] = torch.constant.none
    // CHECK: %[[false:.*]] = torch.constant.bool false
    // CHECK: %[[x4:.*]] = torch.aten.tensor %[[x3]], %[[none]], %[[none]], %[[false]] : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[4,2],si64>
    // CHECK: return %[[x4]] : !torch.vtensor<[4,2],si64>    
    %0 = torch.vtensor.literal(dense<0> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %int-9223372036854775807 = torch.constant.int -9223372036854775807
    %int-1 = torch.constant.int -1
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %3 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,144,?,?],f32> -> !torch.vtensor<[4],si64>
    %4 = torch.prim.ListConstruct %3, %0 : (!torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>) -> !torch.list<vtensor>
    %5 = torch.aten.cat %4, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[8],si64>
    %6 = torch.prim.ListConstruct %int-1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.view %5, %6 : !torch.vtensor<[8],si64>, !torch.list<int> -> !torch.vtensor<[4,2],si64>
    %8 = torch.aten.slice.Tensor %7, %int0, %int-1, %int-9223372036854775807, %int-1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,2],si64>
    %9 = torch.aten.transpose.int %8, %int0, %int1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4],si64>
    %10 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %11 = torch.aten.view %9, %10 : !torch.vtensor<[2,4],si64>, !torch.list<int> -> !torch.vtensor<[8],si64>
    %12 = torch.aten.index_select %11, %int0, %1 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %13 = torch.aten.item %12 : !torch.vtensor<[1],si64> -> !torch.int
    %14 = torch.aten.index_select %11, %int0, %2 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %15 = torch.aten.item %14 : !torch.vtensor<[1],si64> -> !torch.int
    %16 = torch.prim.ListConstruct %13, %15: (!torch.int, !torch.int) -> !torch.list<int>
    return %7 : !torch.vtensor<[4,2],si64>
}

// -----

// CHECK-LABEL: @pytorch_dynamic_pad_export_slice$prop(
func.func @pytorch_dynamic_pad_export_slice$prop(%arg0: !torch.vtensor<[?,144,?,?],f32>) -> !torch.vtensor<[4,2],si64> {
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[x0:.*]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I2:.*]] = torch.constant.int 2
    // CHECK: %[[x1:.*]] = torch.aten.size.int %arg0, %[[I2]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I3:.*]] = torch.constant.int 3
    // CHECK: %[[x2:.*]] = torch.aten.size.int %arg0, %[[I3]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I0_0:.*]] = torch.constant.int 0
    // CHECK: %[[I0_1:.*]] = torch.constant.int 0
    // CHECK: %[[I0_2:.*]] = torch.constant.int 0
    // CHECK: %[[I0_3:.*]] = torch.constant.int 0
    // CHECK: %[[I144:.*]] = torch.constant.int 144
    // CHECK: %[[x3:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]], %[[I0_2]], %[[I0_3]], %[[x1]], %[[x2]], %[[x0]], %[[I144]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[none:.*]] = torch.constant.none
    // CHECK: %[[false:.*]] = torch.constant.bool false
    // CHECK: %[[x4:.*]] = torch.aten.tensor %[[x3]], %[[none]], %[[none]], %[[false]] : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[4,2],si64>
    // CHECK: return %[[x4]] : !torch.vtensor<[4,2],si64>    
    %0 = torch.vtensor.literal(dense<0> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %int-9223372036854775807 = torch.constant.int -9223372036854775807
    %int-1 = torch.constant.int -1
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %3 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,144,?,?],f32> -> !torch.vtensor<[4],si64>
    %4 = torch.prim.ListConstruct %3, %0 : (!torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>) -> !torch.list<vtensor>
    %5 = torch.aten.cat %4, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[8],si64>
    %6 = torch.prim.ListConstruct %int-1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.view %5, %6 : !torch.vtensor<[8],si64>, !torch.list<int> -> !torch.vtensor<[4,2],si64>
    %8 = torch.aten.slice.Tensor %7, %int0, %int-1, %int-9223372036854775807, %int-1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,2],si64>
    %9 = torch.aten.transpose.int %8, %int0, %int1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4],si64>
    %10 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %11 = torch.aten.view %9, %10 : !torch.vtensor<[2,4],si64>, !torch.list<int> -> !torch.vtensor<[8],si64>
    %12 = torch.aten.index_select %11, %int0, %1 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %13 = torch.aten.item %12 : !torch.vtensor<[1],si64> -> !torch.int
    %14 = torch.aten.index_select %11, %int0, %2 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %15 = torch.aten.item %14 : !torch.vtensor<[1],si64> -> !torch.int
    %16 = torch.prim.ListConstruct %13, %15: (!torch.int, !torch.int) -> !torch.list<int>
    return %8 : !torch.vtensor<[4,2],si64>
}

// -----

// CHECK-LABEL: @pytorch_dynamic_pad_export_transpose$prop(
func.func @pytorch_dynamic_pad_export_transpose$prop(%arg0: !torch.vtensor<[?,144,?,?],f32>) -> !torch.vtensor<[2,4],si64> {
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[DIM0:.*]] = torch.aten.size.int %arg0, %[[I0]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I2:.*]] = torch.constant.int 2
    // CHECK: %[[DIM2:.*]] = torch.aten.size.int %arg0, %[[I2]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I3:.*]] = torch.constant.int 3
    // CHECK: %[[DIM3:.*]] = torch.aten.size.int %arg0, %[[I3]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I0_0:.*]] = torch.constant.int 0
    // CHECK: %[[I0_1:.*]] = torch.constant.int 0
    // CHECK: %[[I0_2:.*]] = torch.constant.int 0
    // CHECK: %[[I0_3:.*]] = torch.constant.int 0
    // CHECK: %[[DIM1:.*]] = torch.constant.int 144
    // CHECK: %[[x3:.*]] = torch.prim.ListConstruct %[[I0_0]], %[[I0_1]], %[[DIM2]], %[[DIM0]], %[[I0_2]], %[[I0_3]], %[[DIM3]], %[[DIM1]] : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    // CHECK: %[[none:.*]] = torch.constant.none
    // CHECK: %[[false:.*]] = torch.constant.bool false
    // CHECK: %[[x4:.*]] = torch.aten.tensor %[[x3]], %[[none]], %[[none]], %[[false]] : !torch.list<int>, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[2,4],si64>
    // CHECK: return %[[x4]] : !torch.vtensor<[2,4],si64>    
    %0 = torch.vtensor.literal(dense<0> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %int-9223372036854775807 = torch.constant.int -9223372036854775807
    %int-1 = torch.constant.int -1
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %3 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,144,?,?],f32> -> !torch.vtensor<[4],si64>
    %4 = torch.prim.ListConstruct %3, %0 : (!torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>) -> !torch.list<vtensor>
    %5 = torch.aten.cat %4, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[8],si64>
    %6 = torch.prim.ListConstruct %int-1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.view %5, %6 : !torch.vtensor<[8],si64>, !torch.list<int> -> !torch.vtensor<[4,2],si64>
    %8 = torch.aten.slice.Tensor %7, %int0, %int-1, %int-9223372036854775807, %int-1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,2],si64>
    %9 = torch.aten.transpose.int %8, %int0, %int1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4],si64>
    %10 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %11 = torch.aten.view %9, %10 : !torch.vtensor<[2,4],si64>, !torch.list<int> -> !torch.vtensor<[8],si64>
    %12 = torch.aten.index_select %11, %int0, %1 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %13 = torch.aten.item %12 : !torch.vtensor<[1],si64> -> !torch.int
    %14 = torch.aten.index_select %11, %int0, %2 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %15 = torch.aten.item %14 : !torch.vtensor<[1],si64> -> !torch.int
    %16 = torch.prim.ListConstruct %13, %15: (!torch.int, !torch.int) -> !torch.list<int>
    return %9 : !torch.vtensor<[2,4],si64>
}

// -----

// CHECK-LABEL: @pytorch_dynamic_pad_export_full(
func.func @pytorch_dynamic_pad_export_full(%arg0: !torch.vtensor<[?,144,?,?],f32>) -> !torch.list<int> {
    // CHECK: %[[I2:.*]] = torch.constant.int 2
    // CHECK: %[[DIM2:.*]] = torch.aten.size.int %arg0, %[[I2]] : !torch.vtensor<[?,144,?,?],f32>, !torch.int -> !torch.int
    // CHECK: %[[I0:.*]] = torch.constant.int 0
    // CHECK: %[[x1:.*]] = torch.prim.ListConstruct %[[DIM2]], %[[I0]] : (!torch.int, !torch.int) -> !torch.list<int>
    // CHECK: return %[[x1]] : !torch.list<int>
    %0 = torch.vtensor.literal(dense<0> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
    %1 = torch.vtensor.literal(dense<2> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %int-9223372036854775807 = torch.constant.int -9223372036854775807
    %int-1 = torch.constant.int -1
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %3 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,144,?,?],f32> -> !torch.vtensor<[4],si64>
    %4 = torch.prim.ListConstruct %3, %0 : (!torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>) -> !torch.list<vtensor>
    %5 = torch.aten.cat %4, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[8],si64>
    %6 = torch.prim.ListConstruct %int-1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.view %5, %6 : !torch.vtensor<[8],si64>, !torch.list<int> -> !torch.vtensor<[4,2],si64>
    %8 = torch.aten.slice.Tensor %7, %int0, %int-1, %int-9223372036854775807, %int-1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,2],si64>
    %9 = torch.aten.transpose.int %8, %int0, %int1 : !torch.vtensor<[4,2],si64>, !torch.int, !torch.int -> !torch.vtensor<[2,4],si64>
    %10 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %11 = torch.aten.view %9, %10 : !torch.vtensor<[2,4],si64>, !torch.list<int> -> !torch.vtensor<[8],si64>
    %12 = torch.aten.index_select %11, %int0, %1 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %13 = torch.aten.item %12 : !torch.vtensor<[1],si64> -> !torch.int
    %14 = torch.aten.index_select %11, %int0, %2 : !torch.vtensor<[8],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %15 = torch.aten.item %14 : !torch.vtensor<[1],si64> -> !torch.int
    %16 = torch.prim.ListConstruct %13, %15: (!torch.int, !torch.int) -> !torch.list<int>
    return %16 : !torch.list<int>
}
