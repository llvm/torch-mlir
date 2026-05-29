// RUN: torch-mlir-opt <%s -convert-mxfp4-scaled-mm-v2-to-tosa -convert-torch-to-tosa -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fp4_scaled_mm_bf16(
// CHECK-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<1x128x128xf4E2M1FN>
// CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<1x128x128xf4E2M1FN>
// CHECK-SAME:    %[[AS:[a-zA-Z0-9_]+]]: tensor<1x128x4xf8E8M0FNU>
// CHECK-SAME:    %[[BS:[a-zA-Z0-9_]+]]: tensor<1x128x4xf8E8M0FNU>
// CHECK-SAME:  ) -> tensor<128x128xbf16>
// CHECK:        %[[MM:.*]] = tosa.matmul_t_block_scaled %[[A]], %[[AS]], %[[B]], %[[BS]]
// CHECK-SAME:     block_size = BLOCK_SIZE_32
// CHECK-SAME:     -> tensor<1x128x128xf32>
// CHECK:        %[[CAST:.*]] = tosa.cast %[[MM]]
// CHECK-SAME:     -> tensor<1x128x128xbf16>
// CHECK:        %[[SHAPE:.*]] = tosa.const_shape
// CHECK-SAME:     dense<128> : tensor<2xindex>
// CHECK:        %[[RESHAPE:.*]] = tosa.reshape %[[CAST]], %[[SHAPE]]
// CHECK-SAME:     -> tensor<128x128xbf16>
// CHECK:        return %[[RESHAPE]] : tensor<128x128xbf16>
func.func @fp4_scaled_mm_bf16(
    %arg0: !torch.vtensor<[128,64],ui8>,
    %arg1: !torch.vtensor<[64,128],ui8>,
    %arg2: !torch.vtensor<[512],f8E8M0FNU>,
    %arg3: !torch.vtensor<[512],f8E8M0FNU>) -> !torch.vtensor<[128,128],bf16> {
  %fp4_dtype = torch.constant.int 45
  %blockwise1x32 = torch.constant.int 3
  %swizzle_32_4_4 = torch.constant.int 1
  %out_dtype = torch.constant.int 15
  %none = torch.constant.none
  %a_view = torch.aten.view.dtype %arg0, %fp4_dtype : !torch.vtensor<[128,64],ui8>, !torch.int -> !torch.vtensor<[128,64],ui8>
  %b_view = torch.aten.view.dtype %arg1, %fp4_dtype : !torch.vtensor<[64,128],ui8>, !torch.int -> !torch.vtensor<[64,128],ui8>
  %a_scale = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %a_recipe = torch.prim.ListConstruct %blockwise1x32 : (!torch.int) -> !torch.list<int>
  %a_swizzle = torch.prim.ListConstruct %swizzle_32_4_4 : (!torch.int) -> !torch.list<int>
  %b_scale = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[512],f8E8M0FNU>) -> !torch.list<vtensor>
  %b_recipe = torch.prim.ListConstruct %blockwise1x32 : (!torch.int) -> !torch.list<int>
  %b_swizzle = torch.prim.ListConstruct %swizzle_32_4_4 : (!torch.int) -> !torch.list<int>
  %0 = torch.operator "torch.aten._scaled_mm_v2.default"(%a_view, %b_view, %a_scale, %a_recipe, %a_swizzle, %b_scale, %b_recipe, %b_swizzle, %none, %out_dtype) : (!torch.vtensor<[128,64],ui8>, !torch.vtensor<[64,128],ui8>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int) -> !torch.vtensor<[128,128],bf16>
  return %0 : !torch.vtensor<[128,128],bf16>
}

// -----

// CHECK-LABEL: func.func @fp4_scaled_mm_f32(
// CHECK-SAME:    %[[A:[a-zA-Z0-9_]+]]: tensor<1x128x128xf4E2M1FN>
// CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]: tensor<1x128x128xf4E2M1FN>
// CHECK-SAME:    %[[AS:[a-zA-Z0-9_]+]]: tensor<1x128x4xf8E8M0FNU>
// CHECK-SAME:    %[[BS:[a-zA-Z0-9_]+]]: tensor<1x128x4xf8E8M0FNU>
// CHECK-SAME:  ) -> tensor<128x128xf32>
// CHECK:        %[[MM:[0-9]+]] = tosa.matmul_t_block_scaled %[[A]], %[[AS]], %[[B]], %[[BS]]
// CHECK-SAME:     -> tensor<1x128x128xf32>
// CHECK-NOT:    tosa.cast
// CHECK:        %[[SHAPE:[0-9]+]] = tosa.const_shape
// CHECK:        %[[RESHAPE:[0-9]+]] = tosa.reshape %[[MM]], %[[SHAPE]]
// CHECK-SAME:     -> tensor<128x128xf32>
// CHECK:        return %[[RESHAPE]] : tensor<128x128xf32>
func.func @fp4_scaled_mm_f32(
    %arg0: !torch.vtensor<[128,64],ui8>,
    %arg1: !torch.vtensor<[64,128],ui8>,
    %arg2: !torch.vtensor<[1,128,4],f8E8M0FNU>,
    %arg3: !torch.vtensor<[1,128,4],f8E8M0FNU>) -> !torch.vtensor<[128,128],f32> {
  %fp4_dtype = torch.constant.int 45
  %blockwise1x32 = torch.constant.int 3
  %swizzle_32_4_4 = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %a_view = torch.aten.view.dtype %arg0, %fp4_dtype : !torch.vtensor<[128,64],ui8>, !torch.int -> !torch.vtensor<[128,64],ui8>
  %b_view = torch.aten.view.dtype %arg1, %fp4_dtype : !torch.vtensor<[64,128],ui8>, !torch.int -> !torch.vtensor<[64,128],ui8>
  %a_scale = torch.prim.ListConstruct %arg2 : (!torch.vtensor<[1,128,4],f8E8M0FNU>) -> !torch.list<vtensor>
  %a_recipe = torch.prim.ListConstruct %blockwise1x32 : (!torch.int) -> !torch.list<int>
  %a_swizzle = torch.prim.ListConstruct %swizzle_32_4_4 : (!torch.int) -> !torch.list<int>
  %b_scale = torch.prim.ListConstruct %arg3 : (!torch.vtensor<[1,128,4],f8E8M0FNU>) -> !torch.list<vtensor>
  %b_recipe = torch.prim.ListConstruct %blockwise1x32 : (!torch.int) -> !torch.list<int>
  %b_swizzle = torch.prim.ListConstruct %swizzle_32_4_4 : (!torch.int) -> !torch.list<int>
  %0 = torch.operator "torch.aten._scaled_mm_v2.default"(%a_view, %b_view, %a_scale, %a_recipe, %a_swizzle, %b_scale, %b_recipe, %b_swizzle, %none, %out_dtype) : (!torch.vtensor<[128,64],ui8>, !torch.vtensor<[64,128],ui8>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.list<vtensor>, !torch.list<int>, !torch.list<int>, !torch.none, !torch.int) -> !torch.vtensor<[128,128],f32>
  return %0 : !torch.vtensor<[128,128],f32>
}
