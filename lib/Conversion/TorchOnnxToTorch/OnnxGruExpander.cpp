#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace mlir::torch::onnx_c {

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

  Value ht_minus_zt = b.create<AtenSubTensorOp>(hTy, ht, zt, cstOne);
  Value H_prev_zt = b.create<AtenMulTensorOp>(hTy, H_prev, zt);
  Value H_new = b.create<AtenAddTensorOp>(hTy, ht_minus_zt, H_prev_zt, cstOne);

  return H_new;
}

GruLayerOutput gru_layer(ImplicitLocOpBuilder &b, Value X, Value initial_h,
                         GruWeights weights, GruActivations activations,
                         bool linear_before_reset, int64_t layout) {
  Location loc = b.getLoc();

  auto xTy = cast<ValueTensorType>(X.getType());
  auto hTy = cast<ValueTensorType>(initial_h.getType());
  int64_t seq_len = (layout == 0) ? xTy.getSizes()[0] : xTy.getSizes()[1];
  int64_t batch_size = (layout == 0) ? xTy.getSizes()[1] : xTy.getSizes()[0];
  int64_t input_size = xTy.getSizes()[2];
  int64_t hidden_size = hTy.getSizes()[2];

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
      (layout == 0) ? SmallVector<int64_t>{seq_len, 1, batch_size, hidden_size}
                    : SmallVector<int64_t>{batch_size, seq_len, 1, hidden_size},
      hTy.getDtype());

  auto YShapeList = b.create<PrimListConstructOp>(
      b.getType<ListType>(intType),
      (layout == 0)
          ? ValueRange({cstSeqLen, cstOne, cstBatchSize, cstHiddenSize})
          : ValueRange({cstBatchSize, cstSeqLen, cstOne, cstHiddenSize}));

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
                      TypeRange({loopIndexType, yTy, hTy}), {loc, loc, loc});

    Value loopIndex = loopBody->getArgument(0);
    Value Y_prev = loopBody->getArgument(1);
    Value H_prev = loopBody->getArgument(2);

    auto XtType = b.getType<ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, xTy.getDtype());

    Value Xt = (layout == 0)
                   ? b.create<AtenSelectIntOp>(XtType, X, cstZero, loopIndex)
                   : b.create<AtenSelectIntOp>(XtType, X, cstOne, loopIndex);

    Value H_new =
        gru_cell(b, Xt, H_prev, weights, activations, linear_before_reset);

    Type hTyUnsqueezed = b.getType<ValueTensorType>(
        (layout == 0) ? llvm::SmallVector<int64_t>{1, batch_size, hidden_size}
                      : llvm::SmallVector<int64_t>{batch_size, 1, hidden_size},
        hTy.getDtype());
    Value H_new_unsqueezed =
        (layout == 0) ? b.create<AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstZero)
                      : b.create<AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstOne);

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

  // Binding arguments
  ValueTensorType yTy, Y_hType;
  if (binder.tensorResultTypeAtIndex(yTy, 0) ||
      binder.tensorResultTypeAtIndex(Y_hType, 1)) {
    return rewriter.notifyMatchFailure(binder.op,
                                       "At least one output must be present");
  }

  Value X, W, R, B, initial_h, sequence_lens;
  if (binder.tensorOperandAtIndex(X, 0) || binder.tensorOperandAtIndex(W, 1) ||
      binder.tensorOperandAtIndex(R, 2))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor");

  if (binder.tensorOperandAtIndex(B, 3))
    B = b.create<AtenZerosOp>(W.getType(), W);

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

  int64_t num_directions = 1 + (direction == "bidirectional");

  // Validations
  auto XShape = xTy.getSizes();
  int64_t batch_size = (layout == 0) ? XShape[1] : XShape[0];
  int64_t input_size = XShape[2];
  if (num_directions != wTy.getSizes()[0] || num_directions != 1 ||
      hidden_size != wTy.getSizes()[1] || wTy.getSizes()[2] != input_size * 3)
    return rewriter.notifyMatchFailure(binder.op, "Invalid input dimensions");

  // Setting up initial_h
  auto hTy = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      xTy.getDtype());

  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    auto intType = b.getType<IntType>();
    Value cstNumDirections =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions));
    Value cstBatchSize =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
    Value cstHiddenSize =
        b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
    Value cstNone = b.create<ConstantNoneOp>();
    Value hShape = b.create<PrimListConstructOp>(
        b.getType<ListType>(intType),
        ValueRange({cstNumDirections, cstBatchSize, cstHiddenSize}));
    Value cstDtype = getDtypeIntValueForType(rewriter, loc, xTy.getDtype());
    initial_h =
        b.create<AtenZerosOp>(hTy, hShape, cstDtype, cstNone, cstNone, cstNone);
  }

  if (binder.tensorOperandAtIndex(sequence_lens, 4))
    sequence_lens = b.create<ConstantNoneOp>();

  float clip;
  if (!binder.f32FloatAttr(clip, "clip", 0.0f))
    return rewriter.notifyMatchFailure(
        binder.op, "Clip not supported (specified with a value of " +
                       std::to_string(clip) + ")");

  int64_t linear_before_reset_int;
  if (binder.s64IntegerAttr(linear_before_reset_int, "linear_before_reset", 0))
    linear_before_reset_int = 0;
  bool linear_before_reset = linear_before_reset_int != 0;
  // Setting up weights
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
  Value initial_h_forward = getDirection(0, initial_h);

  GruWeights weights;

  auto intType = b.getType<IntType>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));
  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value cstTwoHiddenSize = b.create<AtenMulIntOp>(
      cstHiddenSize, b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(2)));
  Value cstThreeHiddenSize = b.create<AtenMulIntOp>(
      cstHiddenSize, b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(3)));

  // W slices
  weights.Wz = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, input_size}, wTy.getDtype()),
      W_forward, cstZero, cstZero, cstHiddenSize, cstOne);
  weights.Wr = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, input_size}, wTy.getDtype()),
      W_forward, cstZero, cstHiddenSize, cstTwoHiddenSize, cstOne);
  weights.Wh = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, input_size}, wTy.getDtype()),
      W_forward, cstZero, cstTwoHiddenSize, cstThreeHiddenSize, cstOne);

  // R slices
  weights.Rz = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, hidden_size}, wTy.getDtype()),
      R_forward, cstZero, cstZero, cstHiddenSize, cstOne);
  weights.Rr = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, hidden_size}, wTy.getDtype()),
      R_forward, cstZero, cstHiddenSize, cstTwoHiddenSize, cstOne);
  weights.Rh = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, hidden_size}, wTy.getDtype()),
      R_forward, cstZero, cstTwoHiddenSize, cstThreeHiddenSize, cstOne);

  // B slices
  weights.Wbz = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstZero, cstHiddenSize, cstOne);
  weights.Wbr = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstHiddenSize, cstTwoHiddenSize, cstOne);
  weights.Wbh = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstTwoHiddenSize, cstThreeHiddenSize, cstOne);
  weights.Rbz = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstThreeHiddenSize,
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(4))),
      cstOne);
  weights.Rbr = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero,
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(4))),
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(5))),
      cstOne);
  weights.Rbh = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero,
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(5))),
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(6))),
      cstOne);
  // Call gru_layer
  GruLayerOutput gruLayerOutput =
      gru_layer(b, X, initial_h_forward, weights, activations,
                linear_before_reset, layout);

  Value Y_unsqueezed =
      (layout == 0) ? b.create<AtenUnsqueezeOp>(yTy, gruLayerOutput.Y, cstOne)
                    : gruLayerOutput.Y;
  auto Y_h_unsqueezed_type = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      cast<ValueTensorType>(gruLayerOutput.Y_h.getType()).getDtype());
  Value Y_h_unsqueezed = b.create<AtenUnsqueezeOp>(Y_h_unsqueezed_type,
                                                   gruLayerOutput.Y_h, cstZero);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
