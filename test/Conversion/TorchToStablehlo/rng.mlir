// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s


// -----

// CHECK-LABEL:   func.func @torch.aten.uniform(
// CHECK-SAME:                                    %[[ARG_0:.*]]: !torch.vtensor<[32,64],f64>) -> !torch.vtensor<[32,64],f64> {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FLOAT_0:.*]] = torch.constant.float 0.000000e+00
// CHECK:           %[[VAL_0:.*]] = torch_c.to_f64 %[[FLOAT_0]]
// CHECK:           %[[FLOAT_1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[VAL_1:.*]] = torch_c.to_f64 %[[FLOAT_1]]
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<[32, 64]> : tensor<2xi64>
// CHECK:           %[[ELEM_0:.*]] = tensor.from_elements %[[VAL_0]] : tensor<1xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[ELEM_0]] : tensor<1xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1xf64>) -> tensor<f64>
// CHECK:           %[[ELEM_1:.*]] = tensor.from_elements %[[VAL_1]] : tensor<1xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.convert %[[ELEM_1]] : tensor<1xf64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xf64>) -> tensor<f64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.rng %[[VAL_4]], %[[VAL_6]], %[[VAL_2]], distribution =  UNIFORM : (tensor<f64>, tensor<f64>, tensor<2xi64>) -> tensor<32x64xf64>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<32x64xf64> -> !torch.vtensor<[32,64],f64>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[32,64],f64>
func.func @torch.aten.uniform(%arg0: !torch.vtensor<[32, 64],f64>) -> !torch.vtensor<[32, 64],f64> {
  %none = torch.constant.none
  %float0 = torch.constant.float 0.0
  %float1 = torch.constant.float 1.0
  %0 = torch.aten.uniform %arg0, %float0, %float1, %none : !torch.vtensor<[32, 64],f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[32, 64],f64>
  return %0 : !torch.vtensor<[32, 64],f64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.randn.generator
// CHECK:          %[[NONE:.*]] = torch.constant.none
// CHECK:          %[[INT32:.*]] = torch.constant.int 32
// CHECK:          %[[INT64:.*]] = torch.constant.int 64
// CHECK:          %[[LIST:.*]] = torch.prim.ListConstruct
// CHECK:          %[[SHAPE:.*]] = stablehlo.constant dense<[32, 64]> : tensor<2xi64>
// CHECK:          %[[VAL_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:          %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:          %[[RNG:.*]] = stablehlo.rng %[[VAL_0]], %[[VAL_1]], %[[SHAPE]], distribution =  NORMAL : (tensor<f64>, tensor<f64>, tensor<2xi64>) -> tensor<32x64xf64>
// CHECK:          %[[RET:.*]] = torch_c.from_builtin_tensor %[[RNG]] : tensor<32x64xf64> -> !torch.vtensor<[32,64],f64>
// CHECK:          return %[[RET]] : !torch.vtensor<[32,64],f64>
func.func @torch.aten.randn.generator() -> !torch.vtensor<[32, 64],f64> {
  %none = torch.constant.none
  %int32 = torch.constant.int 32
  %int64 = torch.constant.int 64
  %size = torch.prim.ListConstruct %int32, %int64 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.randn.generator %size, %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[32, 64], f64>
  return %0 : !torch.vtensor<[32, 64],f64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.randn.generator$f32
// CHECK:          %[[NONE:.*]] = torch.constant.none
// CHECK:          %[[INT32:.*]] = torch.constant.int 32
// CHECK:          %[[INT64:.*]] = torch.constant.int 64
// CHECK:          %[[LIST:.*]] = torch.prim.ListConstruct
// CHECK:          %[[SHAPE:.*]] = stablehlo.constant dense<[32, 64]> : tensor<2xi64>
// CHECK:          %[[VAL_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:          %[[VAL_1:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:          %[[RNG:.*]] = stablehlo.rng %[[VAL_0]], %[[VAL_1]], %[[SHAPE]], distribution =  NORMAL : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<32x64xf32>
// CHECK:          %[[RET:.*]] = torch_c.from_builtin_tensor %[[RNG]] : tensor<32x64xf32> -> !torch.vtensor<[32,64],f32>
// CHECK:          return %[[RET]] : !torch.vtensor<[32,64],f32>
func.func @torch.aten.randn.generator$f32() -> !torch.vtensor<[32, 64],f32> {
  %none = torch.constant.none
  %int32 = torch.constant.int 32
  %int64 = torch.constant.int 64
  %size = torch.prim.ListConstruct %int32, %int64 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.randn.generator %size, %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[32, 64], f32>
  return %0 : !torch.vtensor<[32, 64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.normal_functional(
// CHECK-SAME:                                        %[[ARG_0:.*]]: !torch.vtensor<[32,64],f64>) -> !torch.vtensor<[32,64],f64> {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[FLOAT_0:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_0:.*]] = torch_c.to_f64 %[[FLOAT_0]]
// CHECK:           %[[FLOAT_1:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[VAL_1:.*]] = torch_c.to_f64 %[[FLOAT_1]]
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<[32, 64]> : tensor<2xi64>
// CHECK:           %[[ELEM_0:.*]] = tensor.from_elements %[[VAL_0]] : tensor<1xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.convert %[[ELEM_0]] : tensor<1xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<1xf64>) -> tensor<f64>
// CHECK:           %[[ELEM_1:.*]] = tensor.from_elements %[[VAL_1]] : tensor<1xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.convert %[[ELEM_1]] : tensor<1xf64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xf64>) -> tensor<f64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.rng %[[VAL_4]], %[[VAL_6]], %[[VAL_2]], distribution =  NORMAL : (tensor<f64>, tensor<f64>, tensor<2xi64>) -> tensor<32x64xf64>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<32x64xf64> -> !torch.vtensor<[32,64],f64>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[32,64],f64>
func.func @torch.aten.normal_functional(%arg0: !torch.vtensor<[32, 64], f64>) -> !torch.vtensor<[32, 64], f64> {
  %none = torch.constant.none
  %mean = torch.constant.float 2.0
  %std = torch.constant.float 1.0
  %0 = torch.aten.normal_functional %arg0, %mean, %std, %none : !torch.vtensor<[32, 64], f64>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[32, 64], f64>
  return %0 : !torch.vtensor<[32, 64],f64>
}
