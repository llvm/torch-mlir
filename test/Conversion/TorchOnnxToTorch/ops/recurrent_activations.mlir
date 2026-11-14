// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// Gate activations of onnx.RNN, onnx.LSTM and onnx.GRU, with activation_alpha
// and activation_beta or the defaults of the matching ONNX operators.

// CHECK-LABEL: func.func @test_rnn_hardsigmoid
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 2.500000e-01
// CHECK:         %[[BETA:.*]] = torch.constant.float 7.500000e-01
// CHECK:         %[[SCALED:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[ALPHA]]
// CHECK:         %[[ONE:.*]] = torch.constant.float 1.000000e+00
// CHECK:         %[[SHIFTED:.*]] = torch.aten.add.Scalar %[[SCALED]], %[[BETA]], %[[ONE]]
// CHECK:         %[[ZERO:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.clamp %[[SHIFTED]], %[[ZERO]], %[[ONE]]
func.func @test_rnn_hardsigmoid(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["HardSigmoid"], torch.onnx.activation_alpha = [2.500000e-01 : f32], torch.onnx.activation_beta = [7.500000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_hardsigmoid_defaults
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 0.20000000298023224
// CHECK:         %[[BETA:.*]] = torch.constant.float 5.000000e-01
// CHECK:         %[[SCALED:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[ALPHA]]
// CHECK:         %[[ONE:.*]] = torch.constant.float 1.000000e+00
// CHECK:         %[[SHIFTED:.*]] = torch.aten.add.Scalar %[[SCALED]], %[[BETA]], %[[ONE]]
// CHECK:         %[[ZERO:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.clamp %[[SHIFTED]], %[[ZERO]], %[[ONE]]
func.func @test_rnn_hardsigmoid_defaults(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["HardSigmoid"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_leakyrelu_default
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 0.0099999997764825821
// CHECK:         torch.aten.leaky_relu %{{.*}}, %[[ALPHA]]
func.func @test_rnn_leakyrelu_default(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["LeakyRelu"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_thresholdedrelu_default
// CHECK-DAG:     %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK-DAG:     %[[ZERO:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.threshold %{{.*}}, %[[ALPHA]], %[[ZERO]]
func.func @test_rnn_thresholdedrelu_default(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["ThresholdedRelu"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_affine
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 2.000000e+00
// CHECK:         %[[SCALED:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[ALPHA]]
// CHECK:         %[[BETA:.*]] = torch.constant.float 5.000000e-01
// CHECK:         torch.aten.add.Scalar %[[SCALED]], %[[BETA]],
func.func @test_rnn_affine(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Affine"], torch.onnx.activation_alpha = [2.000000e+00 : f32], torch.onnx.activation_beta = [5.000000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Only alpha is given, so beta is 0, as in onnxruntime.
// CHECK-LABEL: func.func @test_rnn_affine_alpha_only
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 2.000000e+00
// CHECK:         %[[SCALED:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[ALPHA]]
// CHECK:         %[[BETA:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.add.Scalar %[[SCALED]], %[[BETA]],
func.func @test_rnn_affine_alpha_only(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Affine"], torch.onnx.activation_alpha = [2.000000e+00 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_scaledtanh
// CHECK:         %[[BETA:.*]] = torch.constant.float 5.000000e-01
// CHECK:         %[[INNER:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[BETA]]
// CHECK:         %[[TANH:.*]] = torch.aten.tanh %[[INNER]]
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 2.000000e+00
// CHECK:         torch.aten.mul.Scalar %[[TANH]], %[[ALPHA]]
func.func @test_rnn_scaledtanh(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["ScaledTanh"], torch.onnx.activation_alpha = [2.000000e+00 : f32], torch.onnx.activation_beta = [5.000000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_elu
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 1.500000e+00
// CHECK:         %[[ONE:.*]] = torch.constant.float 1.000000e+00
// CHECK:         torch.aten.elu %{{.*}}, %[[ALPHA]], %[[ONE]], %[[ONE]]
func.func @test_rnn_elu(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Elu"], torch.onnx.activation_alpha = [1.500000e+00 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_softsign
// CHECK:         %[[ABS:.*]] = torch.aten.abs %[[PRE:.*]] :
// CHECK:         %[[ONE:.*]] = torch.constant.int 1
// CHECK:         %[[DENOM:.*]] = torch.aten.add.Scalar %[[ABS]], %[[ONE]], %[[ONE]]
// CHECK:         torch.aten.div.Tensor %[[PRE]], %[[DENOM]]
func.func @test_rnn_softsign(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Softsign"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_softplus
// CHECK:         %[[PRE:.*]] = torch.aten.add.Tensor
// CHECK-NEXT:    %[[BETA:.*]] = torch.constant.int 1{{$}}
// CHECK-NEXT:    %[[THRESHOLD:.*]] = torch.constant.int 20
// CHECK-NEXT:    torch.aten.softplus %[[PRE]], %[[BETA]], %[[THRESHOLD]]
func.func @test_rnn_softplus(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Softplus"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// activation_alpha is consumed only by activations that take alpha, so
// LeakyRelu (h) gets the second value: Tanh (g) takes none.
// CHECK-LABEL: func.func @test_lstm_alpha_order
// CHECK-DAG:     %[[LEAKY_ALPHA:.*]] = torch.constant.float 7.500000e-01
// CHECK:         torch.aten.leaky_relu %{{.*}}, %[[LEAKY_ALPHA]]
func.func @test_lstm_alpha_order(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["HardSigmoid", "Tanh", "LeakyRelu"], torch.onnx.activation_alpha = [2.500000e-01 : f32, 7.500000e-01 : f32], torch.onnx.activation_beta = [5.000000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// For a bidirectional LSTM the parameter lists continue across all six
// activations: the reverse f (LeakyRelu) gets the second alpha.
// CHECK-LABEL: func.func @test_lstm_bidirectional_alpha_continues
// CHECK-DAG:     %[[LEAKY_ALPHA:.*]] = torch.constant.float 7.500000e-01
// CHECK:         torch.aten.leaky_relu %{{.*}}, %[[LEAKY_ALPHA]]
func.func @test_lstm_bidirectional_alpha_continues(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[2,12,4],f32>, %arg2: !torch.vtensor<[2,12,3],f32>) -> (!torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:3 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.direction = "bidirectional", torch.onnx.activations = ["HardSigmoid", "Tanh", "Tanh", "LeakyRelu", "Tanh", "Tanh"], torch.onnx.activation_alpha = [2.500000e-01 : f32, 7.500000e-01 : f32], torch.onnx.activation_beta = [5.000000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[2,12,4],f32>, !torch.vtensor<[2,12,3],f32>) -> (!torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>)
  return %0#0, %0#1, %0#2 : !torch.vtensor<[3,2,2,3],f32>, !torch.vtensor<[2,2,3],f32>, !torch.vtensor<[2,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_gru_hardsigmoid_softsign
// CHECK-DAG:     %[[ALPHA:.*]] = torch.constant.float 2.500000e-01
// CHECK:         torch.aten.mul.Scalar %{{.*}}, %[[ALPHA]]
// CHECK:         torch.aten.clamp
// CHECK:         torch.aten.abs
// CHECK:         torch.aten.div.Tensor
func.func @test_gru_hardsigmoid_softsign(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,9,4],f32>, %arg2: !torch.vtensor<[1,9,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["HardSigmoid", "Softsign"], torch.onnx.activation_alpha = [2.500000e-01 : f32], torch.onnx.activation_beta = [5.000000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,9,4],f32>, !torch.vtensor<[1,9,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Without an activations attribute, RNN uses Tanh.
// CHECK-LABEL: func.func @test_rnn_default_activation
// CHECK:         torch.aten.tanh
// CHECK-NOT:     torch.aten.sigmoid
func.func @test_rnn_default_activation(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Without an activations attribute, GRU uses Sigmoid for z and r and Tanh for h.
// CHECK-LABEL: func.func @test_gru_default_activations
// CHECK:         torch.aten.sigmoid
// CHECK:         torch.aten.sigmoid
// CHECK:         torch.aten.tanh
func.func @test_gru_default_activations(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,9,4],f32>, %arg2: !torch.vtensor<[1,9,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,9,4],f32>, !torch.vtensor<[1,9,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// activation_beta is consumed only by activations that take beta, so
// HardSigmoid (g) gets the first value: LeakyRelu (f) takes none.
// CHECK-LABEL: func.func @test_gru_beta_order
// CHECK-DAG:     %[[LEAKY_ALPHA:.*]] = torch.constant.float 1.250000e-01
// CHECK:         torch.aten.leaky_relu %{{.*}}, %[[LEAKY_ALPHA]]
// CHECK:         %[[HS_ALPHA:.*]] = torch.constant.float 2.500000e-01
// CHECK:         %[[HS_BETA:.*]] = torch.constant.float 6.250000e-01
// CHECK:         %[[SCALED:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[HS_ALPHA]]
// CHECK:         %[[SHIFTED:.*]] = torch.aten.add.Scalar %[[SCALED]], %[[HS_BETA]],
// CHECK:         torch.aten.clamp %[[SHIFTED]],
func.func @test_gru_beta_order(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,9,4],f32>, %arg2: !torch.vtensor<[1,9,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["LeakyRelu", "HardSigmoid"], torch.onnx.activation_alpha = [1.250000e-01 : f32, 2.500000e-01 : f32], torch.onnx.activation_beta = [6.250000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,9,4],f32>, !torch.vtensor<[1,9,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// CHECK-LABEL: func.func @test_rnn_elu_default
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 1.000000e+00
// CHECK:         torch.aten.elu %{{.*}}, %[[ALPHA]],
func.func @test_rnn_elu_default(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Elu"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Only beta is given, so alpha is 0, as in onnxruntime.
// CHECK-LABEL: func.func @test_rnn_scaledtanh_beta_only
// CHECK:         %[[BETA:.*]] = torch.constant.float 5.000000e-01
// CHECK:         %[[INNER:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[BETA]]
// CHECK:         %[[TANH:.*]] = torch.aten.tanh %[[INNER]]
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.mul.Scalar %[[TANH]], %[[ALPHA]]
func.func @test_rnn_scaledtanh_beta_only(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["ScaledTanh"], torch.onnx.activation_beta = [5.000000e-01 : f32]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Neither alpha nor beta is given, so both are 0, as in onnxruntime.
// CHECK-LABEL: func.func @test_rnn_affine_no_parameters
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 0.000000e+00
// CHECK:         %[[SCALED:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[ALPHA]]
// CHECK:         %[[BETA:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.add.Scalar %[[SCALED]], %[[BETA]],
func.func @test_rnn_affine_no_parameters(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["Affine"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// -----

// Neither alpha nor beta is given, so both are 0, as in onnxruntime.
// CHECK-LABEL: func.func @test_rnn_scaledtanh_no_parameters
// CHECK:         %[[BETA:.*]] = torch.constant.float 0.000000e+00
// CHECK:         %[[INNER:.*]] = torch.aten.mul.Scalar %{{.*}}, %[[BETA]]
// CHECK:         %[[TANH:.*]] = torch.aten.tanh %[[INNER]]
// CHECK:         %[[ALPHA:.*]] = torch.constant.float 0.000000e+00
// CHECK:         torch.aten.mul.Scalar %[[TANH]], %[[ALPHA]]
func.func @test_rnn_scaledtanh_no_parameters(%arg0: !torch.vtensor<[3,2,4],f32>, %arg1: !torch.vtensor<[1,3,4],f32>, %arg2: !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>) attributes {torch.onnx_meta.opset_version = 22 : si64} {
  %0:2 = torch.operator "onnx.RNN"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 3 : si64, torch.onnx.activations = ["ScaledTanh"]} : (!torch.vtensor<[3,2,4],f32>, !torch.vtensor<[1,3,4],f32>, !torch.vtensor<[1,3,3],f32>) -> (!torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %0#0, %0#1 : !torch.vtensor<[3,1,2,3],f32>, !torch.vtensor<[1,2,3],f32>
}
