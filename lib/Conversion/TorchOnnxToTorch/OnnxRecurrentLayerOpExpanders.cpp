#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace mlir::torch::onnx_c {

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
Value getDirection(ImplicitLocOpBuilder b, int64_t direction, Value input) {
  auto inputType = cast<ValueTensorType>(input.getType());
  auto outputType = cast<ValueTensorType>(inputType.getWithSizesAndDtype(
      llvm::SmallVector<int64_t>{inputType.getSizes().drop_front()},
      inputType.getDtype()));
  auto intType = b.getType<IntType>();
  Value selectDim = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstDirection =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(direction));
  return b.create<AtenSelectIntOp>(outputType, input, selectDim, cstDirection);
}

struct RnnWeights {
  Value Wi;
  Value Ri;
  Value Wbi;
  Value Rbi;
};

struct RnnActivations {
  std::string f;
};

Value rnn_cell(ImplicitLocOpBuilder &b, Value Xt, Value H_prev,
               RnnWeights weights, RnnActivations activations) {
  auto hTy = cast<ValueTensorType>(H_prev.getType());

  auto intType = b.getType<IntType>();
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  Value i_x = b.create<AtenLinearOp>(hTy, Xt, weights.Wi, weights.Wbi);
  Value i_h = b.create<AtenLinearOp>(hTy, H_prev, weights.Ri, weights.Rbi);
  Value i = b.create<AtenAddTensorOp>(hTy, i_x, i_h, cstOne);

  Value H_new = createActivationByName(b, activations.f, i);
  return H_new;
}

struct RnnLayerOutput {
  Value Y;
  Value Y_h;
};

RnnLayerOutput rnn_layer(ImplicitLocOpBuilder &b, Value X, Value initial_h,
                         RnnWeights weights, RnnActivations activations) {
  Location loc = b.getLoc();

  auto xTy = cast<ValueTensorType>(X.getType());
  auto hTy = cast<ValueTensorType>(initial_h.getType());
  int64_t seq_len = xTy.getSizes()[0];
  int64_t batch_size = xTy.getSizes()[1];
  int64_t input_size = xTy.getSizes()[2];
  int64_t hidden_size = hTy.getSizes()[1];

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

  Value maxTripCount =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(seq_len));
  Value loopConditionTrue = b.create<ConstantBoolOp>(true);

  Type loopIndexType = intType;
  auto loop = b.create<PrimLoopOp>(TypeRange({yTy, hTy}), maxTripCount,
                                   loopConditionTrue,
                                   ValueRange({Y_initial, initial_h}));
  {
    OpBuilder::InsertionGuard guard(b);
    Block *loopBody =
        b.createBlock(&loop.getRegion(), loop.getRegion().begin(),
                      TypeRange({
                          loopIndexType,
                          yTy,
                          hTy,
                      }),
                      {loc, loc, loc} // locs for the loop body arguments
        );

    Value loopIndex = loopBody->getArgument(0);
    Value Y_prev = loopBody->getArgument(1);
    Value H_prev = loopBody->getArgument(2);

    auto xTy = cast<ValueTensorType>(X.getType());
    auto XtType = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, xTy.getDtype());

    Value Xt = b.create<AtenSelectIntOp>(XtType, X, cstZero, loopIndex);

    Value H_new = rnn_cell(b, Xt, H_prev, weights, activations);

    Type hTyUnsqueezed = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{1, batch_size, hidden_size}, hTy.getDtype());
    Value H_new_unsqueezed =
        b.create<AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstZero);

    auto loopIndexPlusOne = b.create<AtenAddIntOp>(intType, loopIndex, cstOne);
    Value Y_new =
        b.create<AtenSliceScatterOp>(yTy, Y_prev, H_new_unsqueezed, cstZero,
                                     loopIndex, loopIndexPlusOne, cstOne);

    b.create<PrimLoopConditionOp>(loopConditionTrue,
                                  ValueRange({Y_new, H_new}));
  }
  RnnLayerOutput output;
  output.Y = loop.getResult(0);
  output.Y_h = loop.getResult(1);
  return output;
}

static Value StaticTranspose(ImplicitLocOpBuilder b, Value value, int64_t dim0,
                             int64_t dim1) {
  auto valueTy = cast<ValueTensorType>(value.getType());

  SmallVector<int64_t> valueShape(valueTy.getSizes());
  std::swap(valueShape[dim0], valueShape[dim1]);
  valueTy = b.getType<ValueTensorType>(valueShape, valueTy.getDtype());

  auto intType = b.getType<IntType>();
  Value dim0v = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(dim0));
  Value dim1v = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(dim1));

  return b.create<AtenTransposeIntOp>(valueTy, value, dim0v, dim1v);
}

LogicalResult OnnxRnnExpander(OpBinder binder,
                              ConversionPatternRewriter &rewriter) {
  Location loc = binder.getLoc();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);

  auto intType = b.getType<IntType>();
  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  int64_t num_directions = Torch::kUnknownSize;
  int64_t hidden_size = Torch::kUnknownSize;

  // Attributes
  llvm::SmallVector<std::string> activationsList;
  RnnActivations activations;
  activations.f = "Tanh";
  if (!binder.stringArrayAttr(activationsList, "activations") &&
      activationsList.size() > 0) {
    if (activationsList.size() == 1) {
      activations.f = activationsList[0];
    } else if (activationsList.size() == 2) {
      return rewriter.notifyMatchFailure(
          binder.op, "Bi-directional RNN is not yet supported, yet two "
                     "activation function names are provided");
    } else {
      return rewriter.notifyMatchFailure(
          binder.op, "Unsupported number of activation functions: " +
                         std::to_string(activationsList.size()) +
                         " are provided.");
    }
  }

  std::string direction;
  if (!binder.customOpNameStringAttr(direction, "direction", "forward") &&
      direction != "forward")
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported direction attribute value. "
                                       "Only 'forward' is supported but '" +
                                           direction + "' is provided.");
  num_directions = (direction == "bidirectional") ? 2 : 1;

  // hidden_size is required according to the docs,
  // but if we encounter a model that doesn't have it
  // that we really want to just push through, consider
  // deleting this check and making it infer the hidden size
  if (binder.s64IntegerAttr(hidden_size, "hidden_size"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute hidden_size");

  // Other attributes
  int64_t layout;
  if (binder.s64IntegerAttr(layout, "layout", 0))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported layout attribute type.");

  if (layout < 0 || layout > 1)
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported layout attribute value.");

  // Result types
  ValueTensorType yTy, Y_hType;
  auto hasResult0 = binder.tensorResultTypeAtIndex(yTy, 0);
  auto hasResult1 = binder.tensorResultTypeAtIndex(Y_hType, 1);

  if (hasResult0 && hasResult1) {
    return rewriter.notifyMatchFailure(binder.op,
                                       "At least one output must be present");
  }

  // Inputs
  Value X, W, R, B, initial_h;
  if (binder.tensorOperandAtIndex(X, 0))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor X");
  if (binder.tensorOperandAtIndex(W, 1))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor W");
  if (binder.tensorOperandAtIndex(R, 2))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor R");
  if (binder.tensorOperandAtIndex(B, 3)) {
    // if no b found, set to null and create one later
    B = nullptr;
  }
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    // if no initial_h found, set to null and create one later
    initial_h = nullptr;
  }

  if (layout == 1) {
    X = StaticTranspose(b, X, 0, 1);
    if (initial_h)
      initial_h = StaticTranspose(b, initial_h, 0, 1);
  }

  // validation
  auto xTy = cast<ValueTensorType>(X.getType());
  auto wTy = cast<ValueTensorType>(W.getType());
  auto rTy = cast<ValueTensorType>(R.getType());
  auto wShape = wTy.getSizes();
  auto xShape = xTy.getSizes();
  auto rShape = rTy.getSizes();
  assert(wShape.size() == 3);

  int64_t seq_len = xShape[0];
  int64_t batch_size = xShape[1];
  int64_t x_input_size = xShape[2];

  int64_t w_num_directions = wShape[0];
  int64_t w_hidden_size = wShape[1];
  int64_t w_input_size = wShape[2];

  int64_t r_num_directions = rShape[0];
  if (rShape[1] != rShape[2])
    return rewriter.notifyMatchFailure(
        binder.op,
        "R tensor must be square, but got shape: " + std::to_string(rShape[1]) +
            "x" + std::to_string(rShape[2]));
  int64_t r_hidden_size = rShape[1];

  // validate input size
  if (x_input_size != w_input_size) {
    return rewriter.notifyMatchFailure(
        binder.op, "input_size inferred from shape of X (" +
                       std::to_string(x_input_size) +
                       ") does not match the input_size attribute value (" +
                       std::to_string(w_input_size) + ")");
  }

  // validate hidden size
  if (w_hidden_size != Torch::kUnknownSize && hidden_size != w_hidden_size) {
    return rewriter.notifyMatchFailure(
        binder.op, "hidden_size inferred from shape of W (" +
                       std::to_string(w_hidden_size) +
                       ") does not match the hidden_size attribute value (" +
                       std::to_string(hidden_size) + ")");
  }

  if (r_hidden_size != Torch::kUnknownSize && hidden_size != r_hidden_size) {
    return rewriter.notifyMatchFailure(
        binder.op, "hidden_size inferred from shape of R (" +
                       std::to_string(r_hidden_size) +
                       ") does not match the hidden_size attribute value (" +
                       std::to_string(hidden_size) + ")");
  }

  // validate num directions
  if (w_num_directions != Torch::kUnknownSize &&
      w_num_directions != num_directions) {
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions from shape of W (" +
                       std::to_string(w_num_directions) +
                       ") does not match the direction attribute value (" +
                       direction + ")");
  }

  if (r_num_directions != Torch::kUnknownSize &&
      r_num_directions != num_directions) {
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions from shape of R (" +
                       std::to_string(r_num_directions) +
                       ") does not match the direction attribute value (" +
                       direction + ")");
  }

  if (num_directions != 1) {
    return rewriter.notifyMatchFailure(
        binder.op,
        "Unsupported num_directions. Only 1 is currently supported but " +
            std::to_string(num_directions) + " is provided.");
  }

  // Create B and initial_h if not provided,
  // using same dtype as X
  Value cstXDtype = getDtypeIntValueForType(rewriter, loc, xTy.getDtype());
  if (B == nullptr) {
    SmallVector<int64_t> BShape = {num_directions, 2 * hidden_size};
    SmallVector<Value> BShapeListContents = {
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions)),
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(2 * hidden_size))};
    Value BShapeList = b.create<PrimListConstructOp>(
        b.getType<ListType>(intType), BShapeListContents);
    auto BType = b.getType<ValueTensorType>(BShape, wTy.getDtype());
    B = b.create<Torch::AtenZerosOp>(BType, BShapeList, cstXDtype, cstNone,
                                     cstNone, cstNone);
  }
  if (initial_h == nullptr) {
    SmallVector<int64_t> initial_h_shape = {num_directions, batch_size,
                                            hidden_size};
    SmallVector<Value> initial_h_shape_list_contents = {
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions)),
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size)),
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size))};
    Value initial_h_shape_list = b.create<PrimListConstructOp>(
        b.getType<ListType>(intType), initial_h_shape_list_contents);
    auto initial_h_type =
        b.getType<ValueTensorType>(initial_h_shape, wTy.getDtype());
    initial_h =
        b.create<Torch::AtenZerosOp>(initial_h_type, initial_h_shape_list,
                                     cstXDtype, cstNone, cstNone, cstNone);
  }

  Value W_forward = getDirection(b, 0, W);
  Value R_forward = getDirection(b, 0, R);
  Value B_forward = getDirection(b, 0, B);
  Value initial_h_forward = getDirection(b, 0, initial_h);

  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));

  RnnWeights weights;
  weights.Wi = W_forward;
  weights.Ri = R_forward;
  weights.Wbi = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstZero, cstHiddenSize, cstOne);
  weights.Rbi = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstHiddenSize,
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(2))),
      cstOne);

  RnnLayerOutput rnnLayerOutput =
      rnn_layer(b, X, initial_h_forward, weights, activations);

  auto Y_h_unsqueezed_type = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      cast<ValueTensorType>(rnnLayerOutput.Y_h.getType()).getDtype());
  Value Y_h_unsqueezed = b.create<AtenUnsqueezeOp>(Y_h_unsqueezed_type,
                                                   rnnLayerOutput.Y_h, cstZero);

  auto Y_unsqueezed_type = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{seq_len, num_directions, batch_size,
                                 hidden_size},
      cast<ValueTensorType>(rnnLayerOutput.Y_h.getType()).getDtype());
  Value Y_unsqueezed =
      b.create<AtenUnsqueezeOp>(Y_unsqueezed_type, rnnLayerOutput.Y, cstOne);

  if (layout == 1) {
    Y_h_unsqueezed = StaticTranspose(b, Y_h_unsqueezed, 0, 1);
    Y_unsqueezed = StaticTranspose(b, Y_unsqueezed, 1, 2);
    Y_unsqueezed = StaticTranspose(b, Y_unsqueezed, 0, 1);
  }

  if (!yTy)
    Y_unsqueezed = cstNone;
  if (!Y_hType)
    Y_h_unsqueezed = cstNone;

  rewriter.replaceOp(binder.op, {Y_unsqueezed, Y_h_unsqueezed});
  return success();
}

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

  auto getTensorDimSize = [&](Value tensor, int64_t dim) {
    auto dimVal = b.create<ConstantIntOp>(loc, b.getI64IntegerAttr(dim));
    return b.createOrFold<AtenSizeIntOp>(loc, tensor, dimVal);
  };

  auto hTy = cast<ValueTensorType>(initial_h.getType());
  // these names are snake_case for consistency with onnx.LSTM documentation
  Value seq_len = getTensorDimSize(X, 0);
  Value batch_size = getTensorDimSize(X, 1);
  Value input_size = getTensorDimSize(X, 2);
  int64_t hidden_size = hTy.getSizes()[1];

  auto cTy = hTy;

  auto intType = b.getType<IntType>();

  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));
  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));

  auto yTy = getTensorTypeFromShapeValues({seq_len, batch_size, cstHiddenSize},
                                          hTy.getDtype());
  auto YShapeList = b.create<PrimListConstructOp>(
      b.getType<ListType>(intType),
      ValueRange({seq_len, batch_size, cstHiddenSize}));

  int64_t hDtypeInt =
      static_cast<int64_t>(getScalarTypeForType(hTy.getDtype()));
  Value hDtypeIntVal =
      b.create<ConstantIntOp>(loc, b.getI64IntegerAttr(hDtypeInt));

  Value Y_initial = b.create<AtenZerosOp>(yTy, YShapeList, hDtypeIntVal,
                                          cstNone, cstNone, cstNone);

  // Create a for-like PrimLoopOp.
  Value maxTripCount = seq_len;
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
    auto XtType =
        getTensorTypeFromShapeValues({batch_size, input_size}, xTy.getDtype());

    Value Xt = b.create<AtenSelectIntOp>(XtType, X, cstZero, loopIndex);

    auto [H_new, C_new] =
        lstm_cell(b, Xt, H_prev, C_prev, weights, activations);

    auto hTyUnsqueezed = getTensorTypeFromShapeValues(
        {cstOne, batch_size, cstHiddenSize}, hTy.getDtype());
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
  if (binder.tensorResultTypeAtIndex(yTy, 0) &&
      binder.tensorResultTypeAtIndex(Y_hType, 1) &&
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

  // TODO: add defaults for activation_alpha acticvation_beta attributes

  llvm::SmallVector<std::string> activationsList;
  if (binder.stringArrayAttr(activationsList, "activations"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute; activations");

  if (!binder.customOpNameStringAttr(direction, "direction", "forward") &&
      direction != "forward" && direction != "bidirectional")
    return rewriter.notifyMatchFailure(
        binder.op, "Unsupported direction attribute value. "
                   "Only 'forward' / 'bidrectional' are supported but '" +
                       direction + "' is provided.");
  int64_t num_directions = 1 + (direction == "bidirectional");
  bool isBidirectional = direction == "bidirectional";
  // There can be backward activations too
  //  if backward -> look for 6 atcivations (what happens when only three?)

  int64_t num_activations = activationsList.size();
  if (num_activations != 0 && num_activations != 3 && num_activations != 6) {
    return rewriter.notifyMatchFailure(
        binder.op, "activations must either be empty (default), have 3 elements"
                   " (forward) or, have 6 elements (bidirectional), but " +
                       std::to_string(activationsList.size()) +
                       " are provided.");
  }
  // TODO : Add checks, defaults and fails for inputs - sequence_lens, P and
  // attrs- clip, input_forget, layout

  Value B;
  if (binder.tensorOperandAtIndex(B, 3)) {
    Value none = b.create<ConstantNoneOp>();
    Value cstHiddenx8 = b.create<ConstantIntOp>(
        b.getType<IntType>(), b.getI64IntegerAttr(8 * hidden_size));
    Value cstNumDir = b.create<ConstantIntOp>(
        b.getType<IntType>(), b.getI64IntegerAttr(num_directions));
    auto BType = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{num_directions, 8 * hidden_size},
        cast<ValueTensorType>(W.getType()).getDtype());
    Value zerosShapeList = b.create<PrimListConstructOp>(
        b.getType<ListType>(b.getType<IntType>()),
        SmallVector<Value>{cstNumDir, cstHiddenx8});
    B = b.create<AtenZerosOp>(BType, zerosShapeList, none, none, none, none);
  }

  LstmActivations activations, activationsRev;
  // Default case (both forward and reverse)
  activations.f = "Sigmoid";
  activations.g = "Tanh";
  activations.h = "Tanh";
  activationsRev.f = "Sigmoid";
  activationsRev.g = "Tanh";
  activationsRev.h = "Tanh";

  // forward only (also to be added for bidirectional case)
  if (num_activations >= 3) {
    activations.f = activationsList[0];
    activations.g = activationsList[1];
    activations.h = activationsList[2];
  }

  // bidirectional
  if (num_activations == 6) {
    activationsRev.f = activationsList[3];
    activationsRev.g = activationsList[4];
    activationsRev.h = activationsList[5];
  }

  float clip;
  if (!binder.f32FloatAttr(clip, "clip", 0.0) && clip != 0.0)
    return rewriter.notifyMatchFailure(binder.op,
                                       "clip attribute not supported");

  int64_t input_forget;
  if (!binder.s64IntegerAttr(input_forget, "input_forget", 0) &&
      input_forget != 0)
    return rewriter.notifyMatchFailure(
        binder.op, "only input_forget = 0 supported. Got input_forgt = " +
                       std::to_string(input_forget));

  int64_t layout;
  if (!binder.s64IntegerAttr(layout, "layout", 0) && layout != 0 && layout != 1)
    return rewriter.notifyMatchFailure(
        binder.op, "invalid value of layout attribute, expecting 0 / 1 got " +
                       std::to_string(layout));

  Value seqLen = getTensorDimSize(rewriter, X, layout == 0 ? 0 : 1);
  Value batchSize = getTensorDimSize(rewriter, X, layout == 0 ? 1 : 0);

  int64_t x_input_size = xTy.getSizes()[2];
  int64_t w_input_size = wTy.getSizes()[2];
  int64_t input_size = w_input_size;
  if (num_directions != wTy.getSizes()[0])
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions (" + std::to_string(num_directions) +
                       ") does not match the first dimension of wTy (" +
                       std::to_string(wTy.getSizes()[0]) + ")");

  if (4 * hidden_size != wTy.getSizes()[1])
    return rewriter.notifyMatchFailure(
        binder.op, "4 times hidden_size (" + std::to_string(4 * hidden_size) +
                       ") does not match the second dimension of wTy (" +
                       std::to_string(wTy.getSizes()[1]) + ")");
  if (x_input_size != Torch::kUnknownSize) {
    if (w_input_size != x_input_size)
      return rewriter.notifyMatchFailure(
          binder.op, "The input_size of wTy (" + std::to_string(w_input_size) +
                         ") does not match input_size of xTY (" +
                         std::to_string(x_input_size) + ")");

  } else {
    Value x_input_size = Torch::getTensorDimSize(rewriter, X, 2);
    Value w_input_size =
        b.create<ConstantIntOp>(loc, b.getI64IntegerAttr(wTy.getSizes()[2]));

    auto eq = b.create<AtenEqIntOp>(loc, x_input_size, w_input_size);
    rewriter.create<RuntimeAssertOp>(
        loc, eq, rewriter.getStringAttr("The input_size of W must equal X."));
  }

  Value W_forward = getDirection(b, 0, W);
  Value R_forward = getDirection(b, 0, R);
  Value B_forward = getDirection(b, 0, B);

  Value W_reverse, R_reverse, B_reverse;
  if (isBidirectional) {
    W_reverse = getDirection(b, 1, W);
    R_reverse = getDirection(b, 1, R);
    B_reverse = getDirection(b, 1, B);
  }

  auto intType = b.getType<IntType>();

  Value cstNumDirections =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions));
  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  auto hTy = getTensorTypeFromShapeValues(
      {cstNumDirections, batchSize, cstHiddenSize}, xTy.getDtype());
  Value hShape = b.create<PrimListConstructOp>(
      b.getType<ListType>(intType),
      ValueRange({cstNumDirections, batchSize, cstHiddenSize}));

  Value cstDtype = getDtypeIntValueForType(rewriter, loc, xTy.getDtype());

  Value initial_h;
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    // default created for layout 0
    initial_h =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  } else {
    if (layout == 1)
      initial_h = StaticTranspose(b, initial_h, 0, 1);
  }

  Value initial_c;
  if (binder.tensorOperandAtIndex(initial_c, 6)) {
    // default created for layout 0
    initial_c =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  } else {
    if (layout == 1)
      initial_c = StaticTranspose(b, initial_c, 0, 1);
  }

  // convert X from layout 1 to layout 0
  if (layout == 1)
    X = StaticTranspose(b, X, 0, 1);

  // X, initial_h, initial_c are now in layout 0

  Value initial_h_forward = getDirection(b, 0, initial_h);
  Value initial_c_forward = getDirection(b, 0, initial_c);

  Value initial_h_reverse, initial_c_reverse;
  if (isBidirectional) {
    initial_h_reverse = getDirection(b, 1, initial_h);
    initial_c_reverse = getDirection(b, 1, initial_c);
  }

  // Everything hereon is for the forward direction (unless in bidirectional if
  // block), with the direction dimention squeezed out and all inputs in layout
  // 0 format

  LstmWeights weights, weightsRev; // weights and biases

  auto intConst = [&](int64_t val) {
    return b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(val));
  };

  // split B into Wb and Rb
  Value inputWeightsEndIdx = intConst(4 * hidden_size);
  Value recurrentWeightsStartIdx = inputWeightsEndIdx;
  Value recurrentWeightsEndIdx = intConst(8 * hidden_size);
  auto biasType = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size * 4}, wTy.getDtype());
  // forward
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
  Value Wb_reverse, Rb_reverse;
  if (isBidirectional) {
    // reverse
    Wb_reverse = b.create<AtenSliceTensorOp>(biasType,
                                             /*input=*/B_reverse,
                                             /*dim=*/cstZero,
                                             /*start=*/cstZero,
                                             /*end=*/inputWeightsEndIdx,
                                             /*step=*/cstOne);
    Rb_reverse = b.create<AtenSliceTensorOp>(biasType,
                                             /*input=*/B_reverse,
                                             /*dim=*/cstZero,
                                             /*start=*/recurrentWeightsStartIdx,
                                             /*end=*/recurrentWeightsEndIdx,
                                             /*step=*/cstOne);
  }

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

  auto sliceIOFC = [&](std::function<Value(Value, Value, Value)> slicerFunction,
                       Value WoB) {
    // slice into 4 components and return tuple
    return std::make_tuple(
        slicerFunction(cstZero, inputGateWeightsEndIdx, WoB),
        slicerFunction(inputGateWeightsEndIdx, outputGateWeightsEndIdx, WoB),
        slicerFunction(outputGateWeightsEndIdx, forgetGateWeightsEndIdx, WoB),
        slicerFunction(forgetGateWeightsEndIdx, cellGateWeightsEndIdx, WoB));
  };

  auto sliceGateBias = [&](Value startIdx, Value endIdx, Value WoB) {
    return b.create<AtenSliceTensorOp>(gateBiasType, WoB, cstZero, startIdx,
                                       endIdx, cstOne);
  };
  std::tie(weights.Wb_i, weights.Wb_o, weights.Wb_f, weights.Wb_c) =
      sliceIOFC(sliceGateBias, Wb);

  if (isBidirectional)
    std::tie(weightsRev.Wb_i, weightsRev.Wb_o, weightsRev.Wb_f,
             weightsRev.Wb_c) = sliceIOFC(sliceGateBias, Wb_reverse);

  auto sliceGateBiasR = [&](Value startIdx, Value endIdx, Value WoB) {
    return b.create<AtenSliceTensorOp>(gateBiasType, WoB, cstZero, startIdx,
                                       endIdx, cstOne);
  };
  std::tie(weights.Rb_i, weights.Rb_o, weights.Rb_f, weights.Rb_c) =
      sliceIOFC(sliceGateBiasR, Rb);

  if (isBidirectional)
    std::tie(weightsRev.Rb_i, weightsRev.Rb_o, weightsRev.Rb_f,
             weightsRev.Rb_c) = sliceIOFC(sliceGateBiasR, Rb_reverse);

  auto sliceGateWeightsIH = [&](Value startIdx, Value endIdx, Value WoB) {
    return b.create<AtenSliceTensorOp>(gateWeightsTypeIH, WoB, cstZero,
                                       startIdx, endIdx, cstOne);
  };
  std::tie(weights.W_i, weights.W_o, weights.W_f, weights.W_c) =
      sliceIOFC(sliceGateWeightsIH, W_forward);

  if (isBidirectional)
    std::tie(weightsRev.W_i, weightsRev.W_o, weightsRev.W_f, weightsRev.W_c) =
        sliceIOFC(sliceGateWeightsIH, W_reverse);

  auto sliceGateWeightsHH = [&](Value startIdx, Value endIdx, Value WoB) {
    return b.create<AtenSliceTensorOp>(gateWeightsTypeHH, WoB, cstZero,
                                       startIdx, endIdx, cstOne);
  };

  std::tie(weights.R_i, weights.R_o, weights.R_f, weights.R_c) =
      sliceIOFC(sliceGateWeightsHH, R_forward);

  if (isBidirectional)
    std::tie(weightsRev.R_i, weightsRev.R_o, weightsRev.R_f, weightsRev.R_c) =
        sliceIOFC(sliceGateWeightsHH, R_reverse);

  LstmLayerOutput lstmLayerOutput = lstm_layer(
      b, X, initial_h_forward, initial_c_forward, weights, activations);

  Value Y_h_result, Y_c_result, Y_result;

  // if forward (unidirectional) unsqueeze and output
  auto YallDtype =
      cast<ValueTensorType>(lstmLayerOutput.Y_h.getType()).getDtype();
  auto Y_h_Y_c_uni_type = getTensorTypeFromShapeValues(
      {cstOne, batchSize, cstHiddenSize}, YallDtype);

  auto Y_uni_type = getTensorTypeFromShapeValues(
      {seqLen, cstOne, batchSize, cstHiddenSize}, YallDtype);

  auto Y_h_Y_c_res_type = getTensorTypeFromShapeValues(
      {cstNumDirections, batchSize, cstHiddenSize}, YallDtype);

  auto Y_res_type = getTensorTypeFromShapeValues(
      {seqLen, cstNumDirections, batchSize, cstHiddenSize}, YallDtype);

  Value Y_h_forward =
      b.create<AtenUnsqueezeOp>(Y_h_Y_c_uni_type, lstmLayerOutput.Y_h, cstZero);

  Value Y_c_forward =
      b.create<AtenUnsqueezeOp>(Y_h_Y_c_uni_type, lstmLayerOutput.Y_c, cstZero);

  // unsqueeze num_directions dim1 of Y
  // to create the onnx.LSTM output shape [seq_length, num_directions,
  // batch_size, hidden_size]
  Value Y_forward =
      b.create<AtenUnsqueezeOp>(Y_uni_type, lstmLayerOutput.Y, cstOne);

  Y_result = Y_forward;
  Y_h_result = Y_h_forward;
  Y_c_result = Y_c_forward;

  // add bidrectional reverse layer
  // this is just flip X, lstm layer, flip results, stack
  // flip X
  Value dim0, X_reverse, Y_h_reverse, Y_c_reverse, Y_reverse_unflipped,
      Y_reverse, Y_output_list, Y_h_output_list, Y_c_output_list;
  LstmLayerOutput revLstmLayerOutput;
  if (isBidirectional) {
    dim0 = b.create<PrimListConstructOp>(b.getType<ListType>(intType),
                                         SmallVector<Value>{cstZero});
    X_reverse = b.create<AtenFlipOp>(xTy, X, dim0); // flip along seq_len dim
    revLstmLayerOutput =
        lstm_layer(b, X_reverse, initial_h_reverse, initial_c_reverse,
                   weightsRev, activationsRev);

    // unsqueeze  Y_rev, Y_h_rev, Y_c_rev
    Y_h_reverse = b.create<AtenUnsqueezeOp>(Y_h_Y_c_uni_type,
                                            revLstmLayerOutput.Y_h, cstZero);
    Y_c_reverse = b.create<AtenUnsqueezeOp>(Y_h_Y_c_uni_type,
                                            revLstmLayerOutput.Y_c, cstZero);
    Y_reverse_unflipped =
        b.create<AtenUnsqueezeOp>(Y_uni_type, revLstmLayerOutput.Y, cstOne);

    // flip Y_rev on dim 0 [seq_len]
    Y_reverse = b.create<AtenFlipOp>(Y_uni_type, Y_reverse_unflipped, dim0);

    // Concat forward and reverse results on dim 1
    Y_output_list =
        b.create<PrimListConstructOp>(b.getType<ListType>(Y_uni_type),
                                      SmallVector<Value>{Y_forward, Y_reverse});
    Y_result = b.create<AtenCatOp>(Y_res_type, Y_output_list, cstOne);

    // Concat forward and reverse results on dim 0
    Y_h_output_list = b.create<PrimListConstructOp>(
        b.getType<ListType>(Y_h_Y_c_uni_type),
        SmallVector<Value>{Y_h_forward, Y_h_reverse});
    Y_h_result =
        b.create<AtenCatOp>(Y_h_Y_c_res_type, Y_h_output_list, cstZero);

    Y_c_output_list = b.create<PrimListConstructOp>(
        b.getType<ListType>(Y_h_Y_c_uni_type),
        SmallVector<Value>{Y_c_forward, Y_c_reverse});
    Y_c_result =
        b.create<AtenCatOp>(Y_h_Y_c_res_type, Y_c_output_list, cstZero);
  }

  if (layout == 1) {
    // Update Y, Y_h, Y_c results to layout 1
    Y_result = StaticTranspose(b, Y_result, 1, 2);
    Y_result = StaticTranspose(b, Y_result, 0, 1);
    Y_h_result = StaticTranspose(b, Y_h_result, 0, 1);
    Y_c_result = StaticTranspose(b, Y_c_result, 0, 1);
  }

  // Only add outputs specified in onnx output node
  SmallVector<Value> actualOutputs = {Y_result, Y_h_result, Y_c_result},
                     outputs;
  ValueTensorType resTy;
  for (int i = 0; i < binder.getNumResults(); ++i) {
    if (failed(binder.tensorResultTypeAtIndex(resTy, i))) {
      outputs.push_back(cstNone);
    } else {
      outputs.push_back(actualOutputs[i]);
    }
  }

  rewriter.replaceOp(binder.op, outputs);
  return success();
}

// W[zrh] - W parameter weight matrix for update, reset, and hidden gates
// R[zrh] - R recurrence weight matrix for update, reset, and hidden gates
// Wb[zrh] - W bias vectors for update, reset, and hidden gates
// Rb[zrh] - R bias vectors for update, reset, and hidden gates
// backwards currently not supported

struct GruWeights {
  Value Wz;
  Value Wr;
  Value Wh;
  Value Rz;
  Value Rr;
  Value Rh;
  Value Wbz;
  Value Wbr;
  Value Wbh;
  Value Rbz;
  Value Rbr;
  Value Rbh;
};

struct GruLayerOutput {
  Value Y;
  Value Y_h;
};

struct GruActivations {
  std::string f;
  std::string g;
};

Value gru_cell(ImplicitLocOpBuilder &b, Value Xt, Value H_prev,
               GruWeights weights, GruActivations activations,
               bool linear_before_reset) {
  auto hTy = cast<ValueTensorType>(H_prev.getType());

  auto intType = b.getType<IntType>();
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  Value z_w = b.create<AtenLinearOp>(hTy, Xt, weights.Wz, weights.Wbz);
  Value z_r = b.create<AtenLinearOp>(hTy, H_prev, weights.Rz, weights.Rbz);
  Value z_pre = b.create<AtenAddTensorOp>(hTy, z_w, z_r, cstOne);
  Value zt = createActivationByName(b, activations.f, z_pre);

  Value r_w = b.create<AtenLinearOp>(hTy, Xt, weights.Wr, weights.Wbr);
  Value r_r = b.create<AtenLinearOp>(hTy, H_prev, weights.Rr, weights.Rbr);
  Value r_pre = b.create<AtenAddTensorOp>(hTy, r_w, r_r, cstOne);
  Value rt = createActivationByName(b, activations.f, r_pre);

  Value h_w = b.create<AtenLinearOp>(hTy, Xt, weights.Wh, weights.Wbh);
  Value h_r;
  if (linear_before_reset) {
    // when linear_before_reset = 1, multiply r with H_prev to reset
    // before applying linear layer
    Value h_linear =
        b.create<AtenLinearOp>(hTy, H_prev, weights.Rh, weights.Rbh);
    h_r = b.create<AtenMulTensorOp>(hTy, h_linear, rt);
  } else {
    // otherwise, multiply first and then apply linear layer
    Value h_reset = b.create<AtenMulTensorOp>(hTy, H_prev, rt);
    h_r = b.create<AtenLinearOp>(hTy, h_reset, weights.Rh, weights.Rbh);
  }
  Value h_pre = b.create<AtenAddTensorOp>(hTy, h_w, h_r, cstOne);
  Value ht = createActivationByName(b, activations.g, h_pre);

  // Create a constant tensor filled with ones, matching the shape of zt
  Value cstNone = b.create<ConstantNoneOp>();
  int64_t typeInt = (int64_t)getScalarTypeForType(hTy.getDtype());
  Value dtype = b.create<ConstantIntOp>(b.getI64IntegerAttr(typeInt));
  Value ones = b.create<Torch::AtenOnesLikeOp>(
      hTy, zt, dtype, /*layout=*/cstNone,
      /*device=*/cstNone, /*pin_memory=*/cstNone, /*memory_format=*/cstNone);

  Value one_minus_zt = b.create<AtenSubTensorOp>(hTy, ones, zt, cstOne);
  Value ht_scaled = b.create<AtenMulTensorOp>(hTy, one_minus_zt, ht);
  Value H_prev_zt = b.create<AtenMulTensorOp>(hTy, H_prev, zt);
  Value H_new = b.create<AtenAddTensorOp>(hTy, ht_scaled, H_prev_zt, cstOne);

  return H_new;
}

GruLayerOutput gru_layer(ImplicitLocOpBuilder &b, Value X, Value initial_h,
                         GruWeights weights, GruActivations activations,
                         bool linear_before_reset) {
  Location loc = b.getLoc();

  auto xTy = cast<ValueTensorType>(X.getType());
  auto hTy = cast<ValueTensorType>(initial_h.getType());

  // Get sizes and store them in intermediate variables
  auto xTySizes = xTy.getSizes();
  auto hTySizes = hTy.getSizes();

  int64_t seq_len = xTySizes[0];
  int64_t batch_size = xTySizes[1];
  int64_t input_size = xTySizes[2];
  int64_t hidden_size = hTySizes[1];

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
  Value hDtypeIntVal = b.create<ConstantIntOp>(b.getI64IntegerAttr(hDtypeInt));

  Value Y_initial = b.create<AtenZerosOp>(yTy, YShapeList, hDtypeIntVal,
                                          cstNone, cstNone, cstNone);

  Value maxTripCount = cstSeqLen;
  Value loopConditionTrue = b.create<ConstantBoolOp>(true);

  Type loopIndexType = intType;

  auto loop = b.create<PrimLoopOp>(TypeRange({yTy, hTy}), maxTripCount,
                                   loopConditionTrue,
                                   ValueRange({Y_initial, initial_h}));

  {
    OpBuilder::InsertionGuard guard(b);
    Block *loopBody =
        b.createBlock(&loop.getRegion(), loop.getRegion().begin(),
                      TypeRange({loopIndexType, yTy, hTy}), {loc, loc, loc});

    Value loopIndex = loopBody->getArgument(0);
    Value Y_prev = loopBody->getArgument(1);
    Value H_prev = loopBody->getArgument(2);

    auto XtType = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, xTy.getDtype());

    Value Xt = b.create<AtenSelectIntOp>(XtType, X, cstZero, loopIndex);

    Value H_new =
        gru_cell(b, Xt, H_prev, weights, activations, linear_before_reset);

    Type hTyUnsqueezed = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{1, batch_size, hidden_size}, hTy.getDtype());
    Value H_new_unsqueezed =
        b.create<AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstZero);

    auto loopIndexPlusOne = b.create<AtenAddIntOp>(intType, loopIndex, cstOne);
    Value Y_new =
        b.create<AtenSliceScatterOp>(yTy, Y_prev, H_new_unsqueezed, cstZero,
                                     loopIndex, loopIndexPlusOne, cstOne);

    b.create<PrimLoopConditionOp>(loopConditionTrue,
                                  ValueRange({Y_new, H_new}));
  }

  GruLayerOutput output;
  output.Y = loop.getResult(0);
  output.Y_h = loop.getResult(1);

  return output;
}

LogicalResult OnnxGruExpander(OpBinder binder,
                              ConversionPatternRewriter &rewriter) {
  Location loc = binder.getLoc();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);

  auto intType = b.getType<IntType>();
  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  // Binding arguments
  ValueTensorType yTy, Y_hType;
  if (binder.tensorResultTypeAtIndex(yTy, 0) &&
      binder.tensorResultTypeAtIndex(Y_hType, 1)) {
    return rewriter.notifyMatchFailure(binder.op,
                                       "At least one output must be present");
  }

  Value X, W, R, B, initial_h, sequence_lens;
  if (binder.tensorOperandAtIndex(X, 0) || binder.tensorOperandAtIndex(W, 1) ||
      binder.tensorOperandAtIndex(R, 2))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor");

  if (binder.tensorOperandAtIndex(B, 3)) {
    // if no b found, set to null and create one later
    B = nullptr;
  }

  int64_t hidden_size;
  if (binder.s64IntegerAttr(hidden_size, "hidden_size"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute hidden_size");

  auto xTy = cast<ValueTensorType>(X.getType());
  auto wTy = cast<ValueTensorType>(W.getType());

  // Setting up activations
  GruActivations activations;
  activations.f = "Sigmoid";
  activations.g = "Tanh";

  llvm::SmallVector<std::string> activationsList;
  if (!binder.stringArrayAttr(activationsList, "activations") &&
      activationsList.size() == 2) {
    activations.f = activationsList[0];
    activations.g = activationsList[1];
  } else if (activationsList.size() > 0) {
    return rewriter.notifyMatchFailure(
        binder.op, "Unsupported number of activation functions");
  }

  // Other attributes
  int64_t layout;
  if (binder.s64IntegerAttr(layout, "layout", 0))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported layout attribute type.");

  std::string direction;
  if (!binder.customOpNameStringAttr(direction, "direction", "forward") &&
      direction != "forward")
    return rewriter.notifyMatchFailure(binder.op,
                                       "Unsupported direction attribute value");

  int64_t num_directions = direction == "bidirectional" ? 2 : 1;
  // Validations
  auto XShape = xTy.getSizes();
  int64_t batch_size = (layout == 0) ? XShape[1] : XShape[0];
  int64_t seq_len = (layout == 0) ? XShape[0] : XShape[1];
  int64_t input_size = XShape[2];

  std::ostringstream oss;

  if (num_directions != 1) {
    oss << "Expected num_directions to be 1, but got " << num_directions
        << ". ";
  }

  if (hidden_size * 3 != wTy.getSizes()[1]) {
    oss << "Expected dim 1 of W to be the same as 3*hidden_size "
        << 3 * hidden_size << ", but got " << wTy.getSizes()[1] << ". ";
  }

  if (wTy.getSizes()[2] != input_size) {
    oss << "Expected wTy.getSizes()[2] to be " << input_size << ", but got "
        << wTy.getSizes()[2] << ". ";
  }

  if (!oss.str().empty()) {
    return rewriter.notifyMatchFailure(binder.op, oss.str());
  }

  // Setting up initial_h
  auto hTy = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      xTy.getDtype());

  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    Value cstNumDirections =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions));
    Value cstBatchSize =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
    Value cstHiddenSize =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
    Value hShape = b.create<PrimListConstructOp>(
        b.getType<ListType>(intType),
        ValueRange({cstNumDirections, cstBatchSize, cstHiddenSize}));
    Value cstDtype = getDtypeIntValueForType(rewriter, loc, xTy.getDtype());
    initial_h =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  } else {
    if (layout == 1) {
      initial_h = StaticTranspose(b, initial_h, 0, 1);
    }
  }

  if (binder.tensorOperandAtIndex(sequence_lens, 4))
    sequence_lens = b.create<ConstantNoneOp>();

  float clip;
  if (!binder.f32FloatAttr(clip, "clip") && clip != 0.0f)
    return rewriter.notifyMatchFailure(
        binder.op, "Clip not supported (specified with a value of " +
                       std::to_string(clip) + ")");

  int64_t linear_before_reset_int;
  if (binder.s64IntegerAttr(linear_before_reset_int, "linear_before_reset", 0))
    linear_before_reset_int = 0;
  bool linear_before_reset = linear_before_reset_int != 0;

  // fill in B
  Value cstXDtype = getDtypeIntValueForType(rewriter, loc, xTy.getDtype());
  if (B == nullptr) {
    SmallVector<int64_t> BShape = {num_directions, 6 * hidden_size};
    SmallVector<Value> BShapeListContents = {
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions)),
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(6 * hidden_size))};
    Value BShapeList = b.create<PrimListConstructOp>(
        b.getType<ListType>(intType), BShapeListContents);
    auto BType = b.getType<ValueTensorType>(BShape, wTy.getDtype());
    B = b.create<Torch::AtenZerosOp>(BType, BShapeList, cstXDtype, cstNone,
                                     cstNone, cstNone);
  }

  Value W_forward = getDirection(b, 0, W);
  Value R_forward = getDirection(b, 0, R);
  Value B_forward = getDirection(b, 0, B);
  Value initial_h_forward = getDirection(b, 0, initial_h);

  GruWeights weights;

  // Slice a tensor into numSlices slices of size sliceSize
  // This is used for slicing the weights & biases into the individual gates
  auto sliceTensor = [&](Value tensor, int64_t sliceSize, int64_t numSlices,
                         ValueTensorType sliceType) {
    SmallVector<Value> slices;
    for (int64_t i = 0; i < numSlices; ++i) {
      Value start =
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(i * sliceSize));
      Value end = b.create<ConstantIntOp>(
          intType, b.getI64IntegerAttr((i + 1) * sliceSize));

      Value slice = b.create<AtenSliceTensorOp>(sliceType, tensor,
                                                cstZero, // dim to slice on
                                                start, end,
                                                cstOne // step
      );

      slices.push_back(slice);
    }
    return slices;
  };

  // Slice W
  auto wSliceType = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, input_size}, wTy.getDtype());
  auto W_slices = sliceTensor(W_forward, hidden_size, 3, wSliceType);
  std::tie(weights.Wz, weights.Wr, weights.Wh) =
      std::make_tuple(W_slices[0], W_slices[1], W_slices[2]);

  // Slice R
  auto rSliceType = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, hidden_size}, wTy.getDtype());
  auto R_slices = sliceTensor(R_forward, hidden_size, 3, rSliceType);
  std::tie(weights.Rz, weights.Rr, weights.Rh) =
      std::make_tuple(R_slices[0], R_slices[1], R_slices[2]);

  // Slice B
  auto bSliceType = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size}, wTy.getDtype());
  auto B_slices = sliceTensor(B_forward, hidden_size, 6, bSliceType);
  std::tie(weights.Wbz, weights.Wbr, weights.Wbh, weights.Rbz, weights.Rbr,
           weights.Rbh) =
      std::make_tuple(B_slices[0], B_slices[1], B_slices[2], B_slices[3],
                      B_slices[4], B_slices[5]);

  // Process inputs based on layout
  if (layout == 1) {
    X = StaticTranspose(b, X, 0, 1);
  }

  // Weights and biases ready. Calling GRU layer to insert the actual ops.
  GruLayerOutput gruLayerOutput = gru_layer(b, X, initial_h_forward, weights,
                                            activations, linear_before_reset);

  // Process outputs based on layout
  Value Y_final;
  if (binder.tensorResultTypeAtIndex(yTy, 0)) {
    Y_final = cstNone;
  } else {
    if (layout == 0) {
      Y_final = b.create<AtenUnsqueezeOp>(yTy, gruLayerOutput.Y, cstOne);
    } else {
      Type yTy_original = b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{seq_len, 1, batch_size, hidden_size},
          yTy.getDtype());
      Y_final =
          b.create<AtenUnsqueezeOp>(yTy_original, gruLayerOutput.Y, cstOne);
      Y_final = StaticTranspose(b, Y_final, 1, 2);
      Y_final = StaticTranspose(b, Y_final, 0, 1);
    }
  }

  Value Y_h_final;
  if (binder.tensorResultTypeAtIndex(Y_hType, 1)) {
    Y_h_final = cstNone;
  } else {
    if (layout == 0) {
      Y_h_final =
          b.create<AtenUnsqueezeOp>(Y_hType, gruLayerOutput.Y_h, cstZero);
    } else {
      Type y_hTy_original = b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{1, batch_size, hidden_size},
          Y_hType.getDtype());
      Y_h_final = b.create<AtenUnsqueezeOp>(y_hTy_original, gruLayerOutput.Y_h,
                                            cstZero);
      Y_h_final = StaticTranspose(b, Y_h_final, 0, 1);
    }
  }

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_final, Y_h_final});
  return success();
}

} // namespace mlir::torch::onnx_c
