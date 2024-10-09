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
    %0 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?],f32> -> !torch.vtensor<[3],si32>
    return %0 : !torch.vtensor<[3],si32>
}

// -----

// CHECK-LABEL: @shape_as_tensor_dim
func.func @shape_as_tensor_dim(%arg0 : !torch.vtensor<[5,?,?],f32>) -> !torch.vtensor<[],si32> {
    // CHECK: %[[FALSE:.+]] = torch.constant.bool false
    // CHECK: %[[NONE:.+]] = torch.constant.none
    // CHECK: %[[INT1:.+]] = torch.constant.int 1
    // CHECK: %[[SZ:.+]] = torch.aten.size.int %arg0, %[[INT1]]
    // CHECK: %[[LIST:.+]] = torch.prim.ListConstruct %[[INT1]]
    // CHECK: %[[TENSOR:.+]] = torch.aten.full %[[LIST]], %[[SZ]], %[[NONE]], %[[NONE]], %[[NONE]], %[[FALSE]]
    // CHECK: return %[[TENSOR]] : !torch.vtensor<[],si32>
    %shape = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?],f32> -> !torch.vtensor<[3],si32>
    %dim = torch.constant.int 0
    %idx = torch.vtensor.literal(dense<1> : tensor<si32>) : !torch.vtensor<[],si32>
    %select = torch.aten.index_select %shape, %dim, %idx : !torch.vtensor<[3],si32>, !torch.int, !torch.vtensor<[],si32> -> !torch.vtensor<[],si32>
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
    return %out : !torch.int
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
    %shape = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[5,?,?,?],f32> -> !torch.vtensor<[4],si32>
    %dim = torch.constant.int 0
    %start = torch.constant.int 1
    %end = torch.constant.int 5
    %step = torch.constant.int 2
    %slice = torch.aten.slice.Tensor %shape, %dim, %start, %end, %step : !torch.vtensor<[4], si32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2], si32>
    return %slice : !torch.vtensor<[2],si32>
}


// -----

// CHECK-LABEL: @view_as_flatten_static
func.func @view_as_flatten_static(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[?,?,1024],f32> {
    // CHECK-DAG:   %[[TWO:.*]] = torch.constant.int 2
    // CHECK-DAG:   %[[FOUR:.*]] = torch.constant.int 4
    // CHECK-DAG:   %[[FLAT:.*]] = torch.aten.flatten.using_ints %arg0, %[[TWO]], %[[FOUR]] : !torch.vtensor<[?,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,1024],f32>
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

// CHECK-LABEL: @view_as_flatten_dynamic
func.func @view_as_flatten_dynamic(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
    // CHECK-DAG:   %[[TWO:.*]] = torch.constant.int 2
    // CHECK-DAG:   %[[FOUR:.*]] = torch.constant.int 4
    // CHECK-DAG:   %[[FLAT:.*]] = torch.aten.flatten.using_ints %arg0, %[[TWO]], %[[FOUR]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
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
    return %14 : !torch.int
}
