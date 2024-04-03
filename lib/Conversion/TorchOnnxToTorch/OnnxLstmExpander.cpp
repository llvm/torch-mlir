#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir::torch::onnx_c {

Value createActivationByName(ImplicitLocOpBuilder &b, StringRef name,
                             Value input) {
  if (name == "Sigmoid")
    return b.create<Torch::AtenSigmoidOp>(input.getType(), input);
  if (name == "Tanh")
    return b.create<Torch::AtenTanhOp>(input.getType(), input);
  if (name == "Relu")
    return b.create<Torch::AtenReluOp>(input.getType(), input);
  llvm_unreachable("Unsupported activation function");
}

/**
 * @struct LstmWeights
 * @brief A structure to hold LSTM weights.
 *
 * Each W_ weight matrix should have shape [hidden_size, input_size].
 * Each R_ weight matrix should have shape [hidden_size, hidden_size].
 * Each bias vector should have shape [4 * hidden_size].
 */
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
/**
 * @brief This function represents a Long Short-Term Memory (LSTM) cell
 * operation.
 *
 * @param b A builder for constructing operations.
 * @param Xt The input sequence. It has a shape of [batch_size, input_size].
 * @param H_prev The previous hidden state. It has a shape of [batch_size,
 * hidden_size].
 * @param C_prev The previous cell state. It has a shape of [batch_size,
 * hidden_size].
 * @param weights The weights for the LSTM cell. See @ref LstmWeights for shapes
 * @param activations The activation functions for the LSTM cell. Members f,g,h
 * correspond to f,g,h in https://onnx.ai/onnx/operators/onnx__LSTM.html
 * @return The state of the LSTM cell after the operation.
 */
LstmCellState lstm_cell(ImplicitLocOpBuilder &b, Value Xt, Value H_prev,
                        Value C_prev, LstmWeights weights,
                        LstmActivations activations) {

  auto intType = b.getType<Torch::IntType>();
  Torch::ValueTensorType hTy = H_prev.getType().cast<Torch::ValueTensorType>();

  Value cstOne =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  // Apply linear/matmul for each gate separately
  // names are consistent with ONNX LSTM documentation
  Value i_x = b.create<Torch::AtenLinearOp>(hTy, Xt, weights.W_i, weights.Wb_i);
  Value i_h =
      b.create<Torch::AtenLinearOp>(hTy, H_prev, weights.R_i, weights.Rb_i);
  Value i = b.create<Torch::AtenAddTensorOp>(hTy, i_x, i_h, cstOne);
  Value i_act = createActivationByName(b, activations.f, i);

  Value o_x = b.create<Torch::AtenLinearOp>(hTy, Xt, weights.W_o, weights.Wb_o);
  Value o_h =
      b.create<Torch::AtenLinearOp>(hTy, H_prev, weights.R_o, weights.Rb_o);
  Value o = b.create<Torch::AtenAddTensorOp>(hTy, o_x, o_h, cstOne);
  Value o_act = createActivationByName(b, activations.f, o);

  Value f_x = b.create<Torch::AtenLinearOp>(hTy, Xt, weights.W_f, weights.Wb_f);
  Value f_h =
      b.create<Torch::AtenLinearOp>(hTy, H_prev, weights.R_f, weights.Rb_f);
  Value f = b.create<Torch::AtenAddTensorOp>(hTy, f_x, f_h, cstOne);
  Value f_act = createActivationByName(b, activations.f, f);

  Value ct_x =
      b.create<Torch::AtenLinearOp>(hTy, Xt, weights.W_c, weights.Wb_c);
  Value ct_h =
      b.create<Torch::AtenLinearOp>(hTy, H_prev, weights.R_c, weights.Rb_c);
  Value ct = b.create<Torch::AtenAddTensorOp>(hTy, ct_x, ct_h, cstOne);
  Value ct_act = createActivationByName(b, activations.g, ct);

  Value C_forget = b.create<Torch::AtenMulTensorOp>(hTy, f_act, C_prev);
  Value C_input = b.create<Torch::AtenMulTensorOp>(hTy, i_act, ct_act);

  LstmCellState newCellState;
  newCellState.C =
      b.create<Torch::AtenAddTensorOp>(hTy, C_forget, C_input, cstOne);
  Value C_new_act = createActivationByName(b, activations.h, newCellState.C);
  newCellState.H = b.create<Torch::AtenMulTensorOp>(hTy, o_act, C_new_act);
  return newCellState;
}

struct LstmLayerOutput {
  Value Y;
  Value Y_h;
  Value Y_c;
};

/**
 * @brief This function implements the LSTM (Long Short-Term Memory) layer
 * operation.
 *
 * The core computation is performed in a loop that iterates over the sequence
 * length. In each iteration, it selects the corresponding input, computes the
 * new hidden state and cell state using the lstm_cell function, and updates the
 * output tensor.
 *
 * @return A struct containing the hidden state historty, final hidden state,
 * and final cell state.
 */
LstmLayerOutput lstm_layer(ImplicitLocOpBuilder &b, Value X, Value initial_h,
                           Value initial_c, LstmWeights weights,
                           LstmActivations activations) {

  Location loc = b.getLoc();

  auto xTy = X.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType hTy =
      initial_h.getType().cast<Torch::ValueTensorType>();
  // these names are snake_case for consistency with onnx.LSTM documentation
  int64_t seq_len = xTy.getSizes()[0];
  int64_t batch_size = xTy.getSizes()[1];
  int64_t input_size = xTy.getSizes()[2];
  int64_t hidden_size = hTy.getSizes()[1];

  Torch::ValueTensorType cTy = hTy;

  auto intType = b.getType<Torch::IntType>();

  Value cstNone = b.create<Torch::ConstantNoneOp>();
  Value cstZero =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(1));
  Value cstSeqLen =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(seq_len));
  Value cstBatchSize =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
  Value cstHiddenSize =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));

  auto yTy = b.getType<Torch::ValueTensorType>(
      SmallVector<int64_t>{seq_len, batch_size, hidden_size}, hTy.getDtype());

  auto YShapeList = b.create<Torch::PrimListConstructOp>(
      b.getType<Torch::ListType>(intType),
      ValueRange({cstSeqLen, cstBatchSize, cstHiddenSize}));

  int hDtypeInt = (int)getScalarTypeForType(hTy.getDtype());
  Value hDtypeIntVal =
      b.create<ConstantIntOp>(loc, b.getI64IntegerAttr(hDtypeInt));

  Value Y_initial = b.create<Torch::AtenZerosOp>(yTy, YShapeList, hDtypeIntVal,
                                                 cstNone, cstNone, cstNone);

  // Create a for-like PrimLoopOp.
  Value maxTripCount =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(seq_len));
  Value cTrue = b.create<Torch::ConstantBoolOp>(true);

  Type loopIndexType = intType;
  auto loop = b.create<Torch::PrimLoopOp>(
      /*results=*/TypeRange({yTy, hTy, cTy}), maxTripCount,
      /*initialCondition=*/cTrue,
      /*iterArgsInit=*/ValueRange({Y_initial, initial_h, initial_c}));
  {
    OpBuilder::InsertionGuard guard(b);
    Block *loopBody = b.createBlock(
        /*parentRegion=*/&loop.getRegion(),
        /*insertionPoint=*/loop.getRegion().begin(),
        /*argumentTypes=*/
        TypeRange({
            loopIndexType, // loop condition
            yTy,
            hTy,
            cTy,
        }),
        /*locations=*/{loc, loc, loc, loc});

    Value loopIndex = loopBody->getArgument(0);
    Value Y_prev = loopBody->getArgument(1);
    Value H_prev = loopBody->getArgument(2);
    Value C_prev = loopBody->getArgument(3);

    Torch::ValueTensorType xTy = X.getType().cast<Torch::ValueTensorType>();
    Torch::ValueTensorType XtType = b.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, xTy.getDtype());

    Value Xt = b.create<Torch::AtenSelectIntOp>(XtType, X, cstZero, loopIndex);

    auto [H_new, C_new] =
        lstm_cell(b, Xt, H_prev, C_prev, weights, activations);

    Type hTyUnsqueezed = b.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{1, batch_size, hidden_size}, hTy.getDtype());
    Value H_new_unsqueezed =
        b.create<Torch::AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstZero);

    auto loopIndexPlusOne =
        b.create<Torch::AtenAddIntOp>(intType, loopIndex, cstOne);
    Value Y_new = b.create<Torch::AtenSliceScatterOp>(
        yTy, Y_prev, H_new_unsqueezed, cstZero, loopIndex, loopIndexPlusOne,
        cstOne);

    b.create<Torch::PrimLoopConditionOp>(
        /*shouldContinue=*/cTrue,
        /*iterArgs=*/ValueRange({Y_new, H_new, C_new}));
  }

  return LstmLayerOutput{.Y = loop.getResult(0),
                         .Y_h = loop.getResult(1),
                         .Y_c = loop.getResult(2)};
}

/**
 * @brief Expands an ONNX LSTM operation into torch ops.
 *
 * This function primarily handles the binding of operands and slicing of the
 * weight matrix. The majority of the lowering process is managed in the
 * lstm_layer and lstm_cell. For the shapes and meanings of the inputs, refer to
 * the ONNX LSTM documentation at:
 * https://onnx.ai/onnx/operators/onnx__LSTM.html
 * The variable names are also consistent with the aforementioned documentation.
 *
 * @param binder The OpBinder object used for binding operands.
 * @param rewriter The ConversionPatternRewriter object used for rewriting
 * patterns.
 * @return LogicalResult indicating the success or failure of the operation.
 */
LogicalResult OnnxLstmExpander(OpBinder binder,
                               ConversionPatternRewriter &rewriter) {
  Location loc = binder.getLoc();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);

  std::string direction;

  Torch::ValueTensorType yTy, Y_hType, Y_cType;
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

  Torch::ValueTensorType xTy = X.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType wTy = W.getType().cast<Torch::ValueTensorType>();
  Value B;
  if (binder.tensorOperandAtIndex(B, 3)) {
    B = b.create<Torch::AtenZerosOp>(W.getType(), W);
  }

  llvm::SmallVector<std::string> activationsList;
  if (binder.stringArrayAttr(activationsList, "activations"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute; activations");

  LstmActivations activations;
  if (activationsList.size() != 3 && activationsList.size() != 0)
    return rewriter.notifyMatchFailure(
        binder.op, "Either empty or 3 activations are supported but " +
                       std::to_string(activationsList.size()) +
                       " are provided.");

  if (activationsList.size() == 3) {
    activations.f = activationsList[0];
    activations.g = activationsList[1];
    activations.h = activationsList[2];
  } else {
    activations.f = "Sigmoid";
    activations.g = "Tanh";
    activations.h = "Tanh";
  }

  if (!binder.customOpNameStringAttr(direction, "direction", "forward") &&
      direction != "forward")
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported direction attribute value. "
                                       "Only 'forward' is supported but '" +
                                           direction + "' is provided.");
  int64_t num_directions = 1 + (direction == "bidirectional");

  assert(num_directions == wTy.getSizes()[0]);
  assert(num_directions == 1);
  assert(4 * hidden_size == wTy.getSizes()[1]);

  auto XShape = xTy.getSizes();
  int64_t batch_size = XShape[1];
  int64_t input_size = XShape[2];
  assert(wTy.getSizes()[2] == input_size);

  // split W, R, B into forward and back
  // W = [W_forward, W_reverse]
  // R = [R_forward, R_reverse]
  // B = [B_forward, B_reverse]
  // helper function for splitting one direction out
  auto getDirection = [&](int64_t direction, Value input) {
    Torch::ValueTensorType inputType =
        input.getType().cast<Torch::ValueTensorType>();
    // output type is input type with dim 0 dropped
    Torch::ValueTensorType outputType =
        inputType
            .getWithSizesAndDtype(
                llvm::SmallVector<int64_t>{inputType.getSizes().drop_front()},
                inputType.getDtype())
            .cast<Torch::ValueTensorType>();
    auto intType = b.getType<Torch::IntType>();
    Value cstZero =
        b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(0));
    Value cstDirection =
        b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(direction));

    return b.create<Torch::AtenSelectIntOp>(outputType, input, cstZero,
                                            cstDirection);
  };

  Value W_forward = getDirection(0, W);
  Value R_forward = getDirection(0, R);
  Value B_forward = getDirection(0, B);

  Torch::ValueTensorType hTy = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      xTy.getDtype());

  auto intType = b.getType<Torch::IntType>();

  // construct a list containing the shape of initial_h and initial_c
  // this is used to check if initial_h and initial_c are provided
  Value cstNumDirections = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(num_directions));
  Value cstBatchSize =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
  Value cstHiddenSize =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value cstNone = b.create<Torch::ConstantNoneOp>();
  Value cstZero =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  Value HShape = b.create<Torch::PrimListConstructOp>(
      b.getType<Torch::ListType>(intType),
      ValueRange({cstNumDirections, cstBatchSize, cstHiddenSize}));

  Value cstDtype =
      Torch::getDtypeIntValueForType(rewriter, loc, xTy.getDtype());

  Value initial_h;
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    initial_h = b.create<Torch::AtenZerosOp>(hTy, HShape, cstDtype, cstNone,
                                             cstNone, cstNone);
  }
  Value initial_c;
  if (binder.tensorOperandAtIndex(initial_c, 6)) {
    initial_c = b.create<Torch::AtenZerosOp>(hTy, HShape, cstDtype, cstNone,
                                             cstNone, cstNone);
  }

  Value initial_h_forward = getDirection(0, initial_h);
  Value initial_c_forward = getDirection(0, initial_c);

  // Everything hereon is for the forward direction, with the direction
  // dimention squeezed out.
  Value hSizeX1 =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value hSizeX2 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(2 * hidden_size));
  Value hSizeX3 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(3 * hidden_size));
  Value hSizeX4 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(4 * hidden_size));
  Value hSizeX8 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(8 * hidden_size));

  Torch::ValueTensorType biasType = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size * 4}, wTy.getDtype());

  Value Wb = b.create<Torch::AtenSliceTensorOp>(
      biasType,
      /*input=*/B_forward, /*dim=*/cstZero, /*start=*/cstZero,
      /*end=*/hSizeX4, /*step=*/cstOne);
  Value Rb = b.create<Torch::AtenSliceTensorOp>(
      biasType,
      /*input=*/B_forward, /*dim=*/cstZero, /*start=*/hSizeX4,
      /*end=*/hSizeX8, /*step=*/cstOne);

  // split W, R, B into individual weight matrices

  Torch::ValueTensorType gateWeightsTypeIH = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, input_size},
      W_forward.getType().cast<Torch::ValueTensorType>().getDtype());

  LstmWeights weights;
  weights.W_i = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeIH, W_forward, cstZero, cstZero, hSizeX1, cstOne);
  weights.W_o = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeIH, W_forward, cstZero, hSizeX1, hSizeX2, cstOne);
  weights.W_f = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeIH, W_forward, cstZero, hSizeX2, hSizeX3, cstOne);
  weights.W_c = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeIH, W_forward, cstZero, hSizeX3, hSizeX4, cstOne);

  Torch::ValueTensorType gateWeightsTypeHH = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, hidden_size},
      R_forward.getType().cast<Torch::ValueTensorType>().getDtype());
  weights.R_i = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeHH, R_forward, cstZero, cstZero, hSizeX1, cstOne);
  weights.R_o = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeHH, R_forward, cstZero, hSizeX1, hSizeX2, cstOne);
  weights.R_f = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeHH, R_forward, cstZero, hSizeX2, hSizeX3, cstOne);
  weights.R_c = b.create<Torch::AtenSliceTensorOp>(
      gateWeightsTypeHH, R_forward, cstZero, hSizeX3, hSizeX4, cstOne);

  Torch::ValueTensorType gateBiasType = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size},
      Wb.getType().cast<Torch::ValueTensorType>().getDtype());
  weights.Wb_i = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                    cstZero, hSizeX1, cstOne);
  weights.Wb_o = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                    hSizeX1, hSizeX2, cstOne);
  weights.Wb_f = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                    hSizeX2, hSizeX3, cstOne);
  weights.Wb_c = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                    hSizeX3, hSizeX4, cstOne);
  weights.Rb_i = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                    cstZero, hSizeX1, cstOne);
  weights.Rb_o = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                    hSizeX1, hSizeX2, cstOne);
  weights.Rb_f = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                    hSizeX2, hSizeX3, cstOne);
  weights.Rb_c = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                    hSizeX3, hSizeX4, cstOne);

  LstmLayerOutput lstmLayerOutput = lstm_layer(
      b, X, initial_h_forward, initial_c_forward, weights, activations);

  if (num_directions != 1) {
    return rewriter.notifyMatchFailure(
        binder.op, "Unsupported num_directions. Only 1 is supported but " +
                       std::to_string(num_directions) + " is provided.");
  } // TODO: support bidirectional LSTM by doing both directions and replacing
    // Unsqueeze with Stack

  Torch::ValueTensorType Y_h_Y_c_unsqueezed_type =
      b.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
          lstmLayerOutput.Y_h.getType()
              .cast<Torch::ValueTensorType>()
              .getDtype());
  Value Y_h_unsqueezed = b.create<Torch::AtenUnsqueezeOp>(
      Y_h_Y_c_unsqueezed_type, lstmLayerOutput.Y_h, cstZero);
  Value Y_c_unsqueezed = b.create<Torch::AtenUnsqueezeOp>(
      Y_h_Y_c_unsqueezed_type, lstmLayerOutput.Y_c, cstZero);

  // unsqueeze num_directions dim1 of Y
  // to create the onnx.LSTM output shape [seq_length, num_directions,
  // batch_size, hidden_size]
  Value Y_unsqueezed =
      b.create<Torch::AtenUnsqueezeOp>(yTy, lstmLayerOutput.Y, cstOne);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed,
                                                 Y_c_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
