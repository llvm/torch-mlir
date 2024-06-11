#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch::Torch;
namespace mlir::torch::onnx_c {
// @struct LstmWeights
// @brief A structure to hold LSTM weights.
//
// Each W_ weight matrix should have shape [hidden_size, input_size].
// Each R_ weight matrix should have shape [hidden_size, hidden_size].
// Each bias vector should have shape [4 * hidden_size].
struct LstmWeights {
  Value W_i, W_o, W_f, W_c;
  Value R_i, R_o, R_f, R_c;
  Value Wb_i, Wb_o, Wb_f, Wb_c;
  Value Rb_i, Rb_o, Rb_f, Rb_c;
};
struct LstmActivations {
  std::string f;
  std::string g;
  std::string h;
};

struct LstmCellState {
  Value H;
  Value C;
};
// This function represents a Long Short-Term Memory (LSTM) cell operation.
//
// @param b A builder for constructing operations.
// @param Xt The input sequence. It has a shape of [batch_size, input_size].
// @param H_prev The previous hidden state. It has a shape of [batch_size,
// hidden_size].
// @param C_prev The previous cell state. It has a shape of [batch_size,
// hidden_size].
// @param weights The weights for the LSTM cell. See @ref LstmWeights for shapes
// @param activations The activation functions for the LSTM cell. Members f,g,h
// correspond to f,g,h in https://onnx.ai/onnx/operators/onnx__LSTM.html
// @return The state of the LSTM cell after the operation.
LstmCellState lstm_cell(ImplicitLocOpBuilder &b, Value Xt, Value H_prev,
                        Value C_prev, LstmWeights weights,
                        LstmActivations activations) {

  auto intType = b.getType<IntType>();
  auto hTy = cast<ValueTensorType>(H_prev.getType());

  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  // Apply linear/matmul for each gate separately
  // names are consistent with ONNX LSTM documentation
  Value i_x = b.create<AtenLinearOp>(hTy, Xt, weights.W_i, weights.Wb_i);
  Value i_h = b.create<AtenLinearOp>(hTy, H_prev, weights.R_i, weights.Rb_i);
  Value i = b.create<AtenAddTensorOp>(hTy, i_x, i_h, cstOne);
  Value i_act = createActivationByName(b, activations.f, i);

  Value o_x = b.create<AtenLinearOp>(hTy, Xt, weights.W_o, weights.Wb_o);
  Value o_h = b.create<AtenLinearOp>(hTy, H_prev, weights.R_o, weights.Rb_o);
  Value o = b.create<AtenAddTensorOp>(hTy, o_x, o_h, cstOne);
  Value o_act = createActivationByName(b, activations.f, o);

  Value f_x = b.create<AtenLinearOp>(hTy, Xt, weights.W_f, weights.Wb_f);
  Value f_h = b.create<AtenLinearOp>(hTy, H_prev, weights.R_f, weights.Rb_f);
  Value f = b.create<AtenAddTensorOp>(hTy, f_x, f_h, cstOne);
  Value f_act = createActivationByName(b, activations.f, f);

  Value ct_x = b.create<AtenLinearOp>(hTy, Xt, weights.W_c, weights.Wb_c);
  Value ct_h = b.create<AtenLinearOp>(hTy, H_prev, weights.R_c, weights.Rb_c);
  Value ct = b.create<AtenAddTensorOp>(hTy, ct_x, ct_h, cstOne);
  Value ct_act = createActivationByName(b, activations.g, ct);

  Value C_forget = b.create<AtenMulTensorOp>(hTy, f_act, C_prev);
  Value C_input = b.create<AtenMulTensorOp>(hTy, i_act, ct_act);

  LstmCellState newCellState;
  newCellState.C = b.create<AtenAddTensorOp>(hTy, C_forget, C_input, cstOne);
  Value C_new_act = createActivationByName(b, activations.h, newCellState.C);
  newCellState.H = b.create<AtenMulTensorOp>(hTy, o_act, C_new_act);
  return newCellState;
}

struct LstmLayerOutput {
  Value Y;
  Value Y_h;
  Value Y_c;
};

// @brief This function implements the LSTM (Long Short-Term Memory) layer
// operation.
//
// The core computation is performed in a loop that iterates over the sequence
// length. In each iteration, it selects the corresponding input, computes the
// new hidden state and cell state using the lstm_cell function, and updates the
// output tensor.
//
// @return A struct containing the hidden state history, final hidden state,
// and final cell state.
LstmLayerOutput lstm_layer(ImplicitLocOpBuilder &b, Value X, Value initial_h,
                           Value initial_c, LstmWeights weights,
                           LstmActivations activations) {

  Location loc = b.getLoc();

  auto xTy = cast<ValueTensorType>(X.getType());
  auto hTy = cast<ValueTensorType>(initial_h.getType());
  // these names are snake_case for consistency with onnx.LSTM documentation
  int64_t seq_len = xTy.getSizes()[0];
  int64_t batch_size = xTy.getSizes()[1];
  int64_t input_size = xTy.getSizes()[2];
  int64_t hidden_size = hTy.getSizes()[1];

  auto cTy = hTy;

  auto intType = b.getType<IntType>();

  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));
  Value cstSeqLen =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(seq_len));
  Value cstBatchSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));

  auto yTy = b.getType<ValueTensorType>(
      SmallVector<int64_t>{seq_len, batch_size, hidden_size}, hTy.getDtype());

  auto YShapeList = b.create<PrimListConstructOp>(
      b.getType<ListType>(intType),
      ValueRange({cstSeqLen, cstBatchSize, cstHiddenSize}));

  int64_t hDtypeInt =
      static_cast<int64_t>(getScalarTypeForType(hTy.getDtype()));
  Value hDtypeIntVal =
      b.create<ConstantIntOp>(loc, b.getI64IntegerAttr(hDtypeInt));

  Value Y_initial = b.create<AtenZerosOp>(yTy, YShapeList, hDtypeIntVal,
                                          cstNone, cstNone, cstNone);

  // Create a for-like PrimLoopOp.
  Value maxTripCount =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(seq_len));
  Value loopConditionTrue = b.create<ConstantBoolOp>(true);

  Type loopIndexType = intType;
  auto loop = b.create<PrimLoopOp>(
      TypeRange({yTy, hTy, cTy}), maxTripCount, loopConditionTrue,
      ValueRange({Y_initial, initial_h, initial_c}));
  {
    OpBuilder::InsertionGuard guard(b);
    Block *loopBody =
        b.createBlock(&loop.getRegion(), loop.getRegion().begin(),
                      TypeRange({
                          loopIndexType,
                          yTy,
                          hTy,
                          cTy,
                      }),
                      {loc, loc, loc, loc} // locs for the loop body arguments
        );

    Value loopIndex = loopBody->getArgument(0);
    Value Y_prev = loopBody->getArgument(1);
    Value H_prev = loopBody->getArgument(2);
    Value C_prev = loopBody->getArgument(3);

    auto xTy = cast<ValueTensorType>(X.getType());
    auto XtType = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, xTy.getDtype());

    Value Xt = b.create<AtenSelectIntOp>(XtType, X, cstZero, loopIndex);

    auto [H_new, C_new] =
        lstm_cell(b, Xt, H_prev, C_prev, weights, activations);

    Type hTyUnsqueezed = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{1, batch_size, hidden_size}, hTy.getDtype());
    Value H_new_unsqueezed =
        b.create<AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstZero);

    auto loopIndexPlusOne = b.create<AtenAddIntOp>(intType, loopIndex, cstOne);
    Value Y_new =
        b.create<AtenSliceScatterOp>(yTy, Y_prev, H_new_unsqueezed, cstZero,
                                     loopIndex, loopIndexPlusOne, cstOne);

    b.create<PrimLoopConditionOp>(loopConditionTrue,
                                  ValueRange({Y_new, H_new, C_new}));
  }
  LstmLayerOutput output;
  output.Y = loop.getResult(0);
  output.Y_h = loop.getResult(1);
  output.Y_c = loop.getResult(2);
  return output;
}
// @brief Expands an ONNX LSTM operation into torch ops.
//
// This function primarily handles the binding of operands and slicing of the
// weight matrix. The majority of the lowering process is managed in the
// lstm_layer and lstm_cell. For the shapes and meanings of the inputs, refer to
// the ONNX LSTM documentation at:
// https://onnx.ai/onnx/operators/onnx__LSTM.html
// The variable names are also consistent with the aforementioned documentation.
//
// This is not e2e tested here but is verified to work numerically downstream in
// SHARK-TestSuite.
//
// TODO: include this test case when the test infrastructure stops initializing
// weights separately for the reference and tested layers.
// @code{.py}
// class LSTMModule(torch.nn.Module):
//     def __init__(self):
//         super().__init__()
//         self.lstm = torch.nn.LSTM(10, 20, 1)
//     @export
//     @annotate_args([
//         None,
//         ([5, 1, 10], torch.float32, True),
//         ([1, 1, 20], torch.float32, True),
//         ([1, 1, 20], torch.float32, True),
//     ])
//     def forward(self, input, h0, c0):
//         return self.lstm(input, (h0, c0))
//
// @register_test_case(module_factory=LSTMModule)
// def LSTMModule_basic(module, tu: TestUtils):
//     inputs = torch.zeros(5,1,10)
//     h0 = torch.zeros(1,1,20)
//     c0 = torch.zeros(1,1,20)
//
//     output, (hn, cn) = module.forward(inputs, h0, c0)
// @endcode
//
// @param binder The OpBinder object used for binding operands.
LogicalResult OnnxLstmExpander(OpBinder binder,
                               ConversionPatternRewriter &rewriter) {
  Location loc = binder.getLoc();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);

  std::string direction;

  ValueTensorType yTy, Y_hType, Y_cType;
  if (binder.tensorResultTypeAtIndex(yTy, 0) ||
      binder.tensorResultTypeAtIndex(Y_hType, 1) ||
      binder.tensorResultTypeAtIndex(Y_cType, 2)) {
    return rewriter.notifyMatchFailure(binder.op,
                                       "At least one outputs must be present");
  }
  Value X;
  if (binder.tensorOperandAtIndex(X, 0))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor X");
  Value W;
  if (binder.tensorOperandAtIndex(W, 1))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor W");
  Value R;
  if (binder.tensorOperandAtIndex(R, 2))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor R");
  int64_t hidden_size;
  if (binder.s64IntegerAttr(hidden_size, "hidden_size"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute hidden_size");

  auto xTy = cast<ValueTensorType>(X.getType());
  auto wTy = cast<ValueTensorType>(W.getType());
  Value B;
  if (binder.tensorOperandAtIndex(B, 3)) {
    B = b.create<AtenZerosOp>(W.getType(), W);
  }

  llvm::SmallVector<std::string> activationsList;
  if (binder.stringArrayAttr(activationsList, "activations"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute; activations");

  LstmActivations activations;
  activations.f = "Sigmoid";
  activations.g = "Tanh";
  activations.h = "Tanh";
  if (activationsList.size() == 3) {
    activations.f = activationsList[0];
    activations.g = activationsList[1];
    activations.h = activationsList[2];
  } else if (activationsList.size() != 0) {
    return rewriter.notifyMatchFailure(
        binder.op, "activations must be empty have 3 elements, but " +
                       std::to_string(activationsList.size()) +
                       " are provided.");
  }

  if (!binder.customOpNameStringAttr(direction, "direction", "forward") &&
      direction != "forward")
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported direction attribute value. "
                                       "Only 'forward' is supported but '" +
                                           direction + "' is provided.");
  int64_t num_directions = 1 + (direction == "bidirectional");

  auto XShape = xTy.getSizes();
  int64_t batch_size = XShape[1];
  int64_t input_size = XShape[2];
  if (num_directions != wTy.getSizes()[0])
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions (" + std::to_string(num_directions) +
                       ") does not match the first dimension of wTy (" +
                       std::to_string(wTy.getSizes()[0]) + ")");
  if (num_directions != 1)
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions (" + std::to_string(num_directions) +
                       ") is not equal to 1");
  if (4 * hidden_size != wTy.getSizes()[1])
    return rewriter.notifyMatchFailure(
        binder.op, "4 times hidden_size (" + std::to_string(4 * hidden_size) +
                       ") does not match the second dimension of wTy (" +
                       std::to_string(wTy.getSizes()[1]) + ")");
  if (wTy.getSizes()[2] != input_size)
    return rewriter.notifyMatchFailure(
        binder.op,
        "The third dimension of wTy (" + std::to_string(wTy.getSizes()[2]) +
            ") does not match input_size (" + std::to_string(input_size) + ")");

  /**
   * @brief Splits the input tensor based on the provided direction.
   *
   * This function is used to split the LSTM parameters (W, R, B) into forward
   * and backward directions. The input tensor is expected to have the forward
   * and backward parameters concatenated along the 0th dimension. The function
   * returns a tensor that contains the parameters for the specified direction.
   *
   * @param direction The direction to split out. 0 for forward, 1 for backward.
   * @param input The input tensor to split.
   * @return The split tensor for the specified direction.
   */
  auto getDirection = [&](int64_t direction, Value input) {
    auto inputType = cast<ValueTensorType>(input.getType());

    // drop 0th dimension
    auto outputType = cast<ValueTensorType>(inputType.getWithSizesAndDtype(
        llvm::SmallVector<int64_t>{inputType.getSizes().drop_front()},
        inputType.getDtype()));

    auto intType = b.getType<IntType>();
    Value selectDim = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
    Value cstDirection =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(direction));
    return b.create<AtenSelectIntOp>(outputType, input, selectDim,
                                     cstDirection);
  };

  Value W_forward = getDirection(0, W);
  Value R_forward = getDirection(0, R);
  Value B_forward = getDirection(0, B);

  auto hTy = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      xTy.getDtype());

  auto intType = b.getType<IntType>();

  Value cstNumDirections =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions));
  Value cstBatchSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  Value hShape = b.create<PrimListConstructOp>(
      b.getType<ListType>(intType),
      ValueRange({cstNumDirections, cstBatchSize, cstHiddenSize}));

  Value cstDtype = getDtypeIntValueForType(rewriter, loc, xTy.getDtype());

  Value initial_h;
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    initial_h =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  }
  Value initial_c;
  if (binder.tensorOperandAtIndex(initial_c, 6)) {
    initial_c =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  }

  Value initial_h_forward = getDirection(0, initial_h);
  Value initial_c_forward = getDirection(0, initial_c);

  if (num_directions != 1) {
    return rewriter.notifyMatchFailure(
        binder.op, "Unsupported num_directions. Only 1 is supported but " +
                       std::to_string(num_directions) + " is provided.");
    // TODO: support bidirectional LSTM by doing both directions and replacing
    // Unsqueeze with Stack
  }
  // Everything hereon is for the forward direction, with the direction
  // dimention squeezed out.

  LstmWeights weights; // weights and biases

  auto intConst = [&](int64_t val) {
    return b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(val));
  };

  // split B into Wb and Rb
  Value inputWeightsEndIdx = intConst(4 * hidden_size);
  Value recurrentWeightsStartIdx = inputWeightsEndIdx;
  Value recurrentWeightsEndIdx = intConst(8 * hidden_size);
  auto biasType = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size * 4}, wTy.getDtype());
  Value Wb = b.create<AtenSliceTensorOp>(biasType,
                                         /*input=*/B_forward,
                                         /*dim=*/cstZero,
                                         /*start=*/cstZero,
                                         /*end=*/inputWeightsEndIdx,
                                         /*step=*/cstOne);
  Value Rb = b.create<AtenSliceTensorOp>(biasType,
                                         /*input=*/B_forward,
                                         /*dim=*/cstZero,
                                         /*start=*/recurrentWeightsStartIdx,
                                         /*end=*/recurrentWeightsEndIdx,
                                         /*step=*/cstOne);

  // gate splitting
  auto gateBiasType = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size},
      cast<ValueTensorType>(Wb.getType()).getDtype());
  auto gateWeightsTypeIH = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, input_size},
      cast<ValueTensorType>(W_forward.getType()).getDtype());
  auto gateWeightsTypeHH = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, hidden_size},
      cast<ValueTensorType>(R_forward.getType()).getDtype());

  Value inputGateWeightsEndIdx = intConst(hidden_size);
  Value outputGateWeightsEndIdx = intConst(2 * hidden_size);
  Value forgetGateWeightsEndIdx = intConst(3 * hidden_size);
  Value cellGateWeightsEndIdx = intConst(4 * hidden_size);

  auto sliceIOFC = [&](std::function<Value(Value, Value)> slicerFunction) {
    // slice into 4 components and return tuple
    return std::make_tuple(
        slicerFunction(cstZero, inputGateWeightsEndIdx),
        slicerFunction(inputGateWeightsEndIdx, outputGateWeightsEndIdx),
        slicerFunction(outputGateWeightsEndIdx, forgetGateWeightsEndIdx),
        slicerFunction(forgetGateWeightsEndIdx, cellGateWeightsEndIdx));
  };

  auto sliceGateBias = [&](Value startIdx, Value endIdx) {
    return b.create<AtenSliceTensorOp>(gateBiasType, Wb, cstZero, startIdx,
                                       endIdx, cstOne);
  };
  std::tie(weights.Wb_i, weights.Wb_o, weights.Wb_f, weights.Wb_c) =
      sliceIOFC(sliceGateBias);

  auto sliceGateBiasR = [&](Value startIdx, Value endIdx) {
    return b.create<AtenSliceTensorOp>(gateBiasType, Rb, cstZero, startIdx,
                                       endIdx, cstOne);
  };
  std::tie(weights.Rb_i, weights.Rb_o, weights.Rb_f, weights.Rb_c) =
      sliceIOFC(sliceGateBiasR);

  auto sliceGateWeightsIH = [&](Value startIdx, Value endIdx) {
    return b.create<AtenSliceTensorOp>(gateWeightsTypeIH, W_forward, cstZero,
                                       startIdx, endIdx, cstOne);
  };
  std::tie(weights.W_i, weights.W_o, weights.W_f, weights.W_c) =
      sliceIOFC(sliceGateWeightsIH);

  auto sliceGateWeightsHH = [&](Value startIdx, Value endIdx) {
    return b.create<AtenSliceTensorOp>(gateWeightsTypeHH, R_forward, cstZero,
                                       startIdx, endIdx, cstOne);
  };
  std::tie(weights.R_i, weights.R_o, weights.R_f, weights.R_c) =
      sliceIOFC(sliceGateWeightsHH);
  LstmLayerOutput lstmLayerOutput = lstm_layer(
      b, X, initial_h_forward, initial_c_forward, weights, activations);

  auto Y_h_Y_c_unsqueezed_type = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      cast<ValueTensorType>(lstmLayerOutput.Y_h.getType()).getDtype());
  Value Y_h_unsqueezed = b.create<AtenUnsqueezeOp>(
      Y_h_Y_c_unsqueezed_type, lstmLayerOutput.Y_h, cstZero);
  Value Y_c_unsqueezed = b.create<AtenUnsqueezeOp>(
      Y_h_Y_c_unsqueezed_type, lstmLayerOutput.Y_c, cstZero);

  // unsqueeze num_directions dim1 of Y
  // to create the onnx.LSTM output shape [seq_length, num_directions,
  // batch_size, hidden_size]
  Value Y_unsqueezed =
      b.create<AtenUnsqueezeOp>(yTy, lstmLayerOutput.Y, cstOne);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed,
                                                 Y_c_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
