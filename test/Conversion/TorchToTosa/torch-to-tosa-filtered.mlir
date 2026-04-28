// RUN: torch-mlir-opt --convert-torch-to-tosa="disabled-patterns=torch.aten.avg_pool2d require-full-tosa-conversion=false" -split-input-file -verify-diagnostics %s | FileCheck %s -check-prefix=DISABLED-POOL

// RUN: torch-mlir-opt --convert-torch-to-tosa="enabled-patterns=torch.aten.permute require-full-tosa-conversion=false" -split-input-file -verify-diagnostics %s | FileCheck %s -check-prefix=ENABLED-PERMUTE


// -----

// COM: Test that disabled pattern (avg_pool2d) is not converted but the other torch op is converted to TOSA
// DISABLED-POOL-LABEL:   func.func @permute_pool2d
// DISABLED-POOL-NOT:     torch.aten.permute
// DISABLED-POOL:         torch.aten.avg_pool2d

// COM: Test that only the enabled pattern (permute) is converted to TOSA
// ENABLED-PERMUTE-LABEL:   func.func @permute_pool2d
// ENABLED-PERMUTE-NOT:     torch.aten.permute
// ENABLED-PERMUTE:         torch.aten.avg_pool2d
func.func @permute_pool2d(%arg0: !torch.vtensor<[1,7,7,512],f32>) -> !torch.vtensor<[1,512,1,1],f32> {
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0, %int3, %int2, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1,7,7,512],f32>, !torch.list<int> -> !torch.vtensor<[1,512,7,7],f32>
    %int7 = torch.constant.int 7
    %false = torch.constant.bool false
    %none = torch.constant.none
    %2 = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.avg_pool2d %1, %2, %3, %4, %false, %false, %none : !torch.vtensor<[1,512,7,7],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,512,1,1],f32>
    return %5 : !torch.vtensor<[1,512,1,1],f32>
}
