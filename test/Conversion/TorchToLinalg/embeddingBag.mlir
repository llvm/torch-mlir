// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL:   func.func @torchAtenEmbeddingBagPaddingIdx
// CHECK:         %[[VAL_0:.*]]: !torch.vtensor<[1000000,64],f32>
// CHECK:         %[[VAL_1:.*]]: !torch.vtensor<[204790],si64>
// CHECK:         %[[VAL_2:.*]]: !torch.vtensor<[2048],si64>
// CHECK:         %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[2048],si64> -> tensor<2048xi64>
// CHECK:         %[[VAL_4:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[204790],si64> -> tensor<204790xi64>
// CHECK:         %[[VAL_5:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1000000,64],f32> -> tensor<1000000x64xf32>
// CHECK-DAG:     %[[VAL_6:.*]] = torch.constant.bool true
// CHECK-DAG:     %[[VAL_7:.*]] = torch.constant.int 0
// CHECK-DAG:     %[[VAL_8:.*]] = torch.constant.bool true
func.func @torchAtenEmbeddingBagPaddingIdx(%weight: !torch.vtensor<[1000000,64],f32>,
                                           %indices: !torch.vtensor<[204790],si64>,
                                           %offsets: !torch.vtensor<[2048],si64>) -> (!torch.vtensor<[2048,64],f32>,
                                                                                     !torch.vtensor<[0],si64>,
                                                                                     !torch.vtensor<[2048],si64>,
                                                                                     !torch.vtensor<[2048],si64>)
 {
    %scale_grad_by_freq = torch.constant.bool true
    %mode = torch.constant.int 0
    %sparse = torch.constant.bool true
    %per_sample_weights = torch.constant.none
    %include_last_offset = torch.constant.bool false
    %padding_idx = torch.constant.none
    %result0, %result1, %result2, %result3 = torch.aten.embedding_bag.padding_idx %weight,
                                                                                  %indices,
                                                                                  %offsets,
                                                                                  %scale_grad_by_freq,
                                                                                  %mode,
                                                                                  %sparse,
                                                                                  %per_sample_weights,
                                                                                  %include_last_offset,
                                                                                  %padding_idx :
                                                                                               !torch.vtensor<[1000000,64],f32>,
                                                                                               !torch.vtensor<[204790],si64>,
                                                                                               !torch.vtensor<[2048],si64>,
                                                                                               !torch.bool,
                                                                                               !torch.int,
                                                                                               !torch.bool,
                                                                                               !torch.none,
                                                                                               !torch.bool,
                                                                                               !torch.none -> !torch.vtensor<[2048,64],f32>,
                                                                                                              !torch.vtensor<[0],si64>,
                                                                                                              !torch.vtensor<[2048],si64>,
                                                                                                              !torch.vtensor<[2048],si64>

    return %result0, %result1, %result2, %result3 : !torch.vtensor<[2048,64],f32>, !torch.vtensor<[0],si64>, !torch.vtensor<[2048],si64>, !torch.vtensor<[2048],si64>
}
