/*
See also:
    test/cpp/torch-mlir/Conversion/TorchOnnxToTorch/OnnxGruExpander.cpp
    test/cpp/torch-mlir/Conversion/TorchOnnxToTorch/OnnxRnnExpander.cpp
    test/cpp/torch-mlir/Conversion/TorchOnnxToTorch/OnnxLstmExpander.cpp
*/

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace mlir::torch::onnx_c {

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

LogicalResult OnnxRnnExpander(OpBinder binder,
                              ConversionPatternRewriter &rewriter) {
  Location loc = binder.getLoc();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);

  auto intType = b.getType<IntType>();
  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));

  ValueTensorType yTy, Y_hType;
  if (binder.tensorResultTypeAtIndex(yTy, 0) ||
      binder.tensorResultTypeAtIndex(Y_hType, 1)) {
    return rewriter.notifyMatchFailure(binder.op,
                                       "At least one output must be present");
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
  auto WShape = wTy.getSizes();
  assert(WShape.size() == 3);
  // use unpacking to get the values of the 3 dimensions
  int64_t num_directions, hidden_size, input_size;
  num_directions = WShape[0];
  hidden_size = WShape[1];
  input_size = WShape[2];
  SmallVector<int64_t, 2> BShape = {num_directions, 2 * hidden_size};
  auto BType = b.getType<ValueTensorType>(BShape, wTy.getDtype());

  Value B;
  if (binder.tensorOperandAtIndex(B, 3)) {
    B = b.create<Torch::AtenZerosOp>(BType, BShape, wTy.getDtype(), cstNone,
                                     cstNone, cstNone);
  }

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
  int64_t num_directions_attr = (direction == "bidirectional") ? 2 : 1;
  if (num_directions == num_directions_attr) {
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions from shape of W (" +
                       std::to_string(num_directions) +
                       ") does not match the direction attribute value (" +
                       std::to_string(num_directions_attr) + ")");
  }

  auto XShape = xTy.getSizes();
  int64_t batch_size = XShape[1];
  if (input_size != XShape[2]) {
    return rewriter.notifyMatchFailure(
        binder.op, "input_size inferred from shape of W (" +
                       std::to_string(input_size) +
                       ") does not match the third dimension of X (" +
                       std::to_string(XShape[2]) + ")");
  }
  if (num_directions != wTy.getSizes()[0])
    return rewriter.notifyMatchFailure(
        binder.op, "num_directions (" + std::to_string(num_directions) +
                       ") does not match the first dimension of wTy (" +
                       std::to_string(wTy.getSizes()[0]) + ")");
  if (num_directions != 1) {
    return rewriter.notifyMatchFailure(
        binder.op, "Unsupported num_directions. Only 1 is supported but " +
                       std::to_string(num_directions) + " is provided.");
  }
  if (hidden_size != wTy.getSizes()[1])
    return rewriter.notifyMatchFailure(
        binder.op, "hidden_size (" + std::to_string(hidden_size) +
                       ") does not match the second dimension of wTy (" +
                       std::to_string(wTy.getSizes()[1]) + ")");
  if (wTy.getSizes()[2] != input_size)
    return rewriter.notifyMatchFailure(
        binder.op,
        "The third dimension of wTy (" + std::to_string(wTy.getSizes()[2]) +
            ") does not match input_size (" + std::to_string(input_size) + ")");

  auto getDirection = [&](int64_t direction, Value input) {
    auto inputType = cast<ValueTensorType>(input.getType());

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

  Value initial_h;
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    initial_h =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  }

  Value initial_h_forward = getDirection(0, initial_h);

  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

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

  Value Y_unsqueezed = b.create<AtenUnsqueezeOp>(yTy, rnnLayerOutput.Y, cstOne);
  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
