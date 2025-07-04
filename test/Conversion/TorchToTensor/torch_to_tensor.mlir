// RUN: torch-mlir-opt <%s -split-input-file -convert-torch-to-tensor | FileCheck %s

// CHECK-LABEL: func.func @test_shape
func.func @test_shape(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3],si64> {
  // CHECK: %[[SHAPE:.+]] = arith.constant dense<[3, 4, 5]> : tensor<3xi64>
  %0 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}

// -----

// CHECK-LABEL: func.func @test_as_strided
func.func @test_as_strided(%arg0: !torch.vtensor<[1,128,1024,192],f32>) -> !torch.vtensor<[1,128,1024,128],f32> {
  %c0_i64 = arith.constant 0 : i64
  %int0 = torch_c.from_i64 %c0_i64
  %c1_i64 = arith.constant 1 : i64
  %int1 = torch_c.from_i64 %c1_i64
  %c128_i64 = arith.constant 128 : i64
  %int128 = torch_c.from_i64 %c128_i64
  %c192_i64 = arith.constant 192 : i64
  %int192 = torch_c.from_i64 %c192_i64
  %c1024_i64 = arith.constant 1024 : i64
  %int1024 = torch_c.from_i64 %c1024_i64
  %c24576_i64 = arith.constant 24576 : i64
  %int24576 = torch_c.from_i64 %c24576_i64
  %c25165824_i64 = arith.constant 25165824 : i64
  %int25165824 = torch_c.from_i64 %c25165824_i64
  %0 = torch.prim.ListConstruct %int1, %int128, %int1024, %int128 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int25165824, %int192, %int24576, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[RESULT:.+]] = tensor.extract_slice %0[0, 0, 0, 0] [1, 128, 1024, 128] [1, 1, 1, 1] : tensor<1x128x1024x192xf32> to tensor<1x128x1024x128xf32>
  %2 = torch.aten.as_strided %arg0, %0, %1, %int0 : !torch.vtensor<[1,128,1024,192],f32>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,128,1024,128],f32>
  return %2 : !torch.vtensor<[1,128,1024,128],f32>
}
