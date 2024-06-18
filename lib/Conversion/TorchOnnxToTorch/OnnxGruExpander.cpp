#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch::Torch;

namespace mlir::torch::onnx_c {

struct GruWeights {
  Value Wi;
  Value Ri;
  Value Wbi;
  Value Rbi;
};

struct GruActivations {
  std::string f;
  std::string g;
};

struct GruLayerOutput {
  Value Y;
  Value Y_h;
};

Value gru_cell(ImplicitLocOpBuilder &b, Value Xt, Value H_prev,
               GruWeights weights, GruActivations activations,
               bool linear_before_reset) {
  auto hTy = cast<ValueTensorType>(H_prev.getType());

  auto intType = b.getType<IntType>();
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  Value z_x = b.create<AtenLinearOp>(hTy, Xt, weights.Wi, weights.Wbi);
  Value z_h = b.create<AtenLinearOp>(hTy, H_prev, weights.Ri, weights.Rbi);
  Value z = b.create<AtenAddTensorOp>(hTy, z_x, z_h, cstOne);
  Value zt = createActivationByName(b, activations.f, z);

  Value r_x = b.create<AtenLinearOp>(hTy, Xt, weights.Wi, weights.Wbi);
  Value r_h = b.create<AtenLinearOp>(hTy, H_prev, weights.Ri, weights.Rbi);
  Value r = b.create<AtenAddTensorOp>(hTy, r_x, r_h, cstOne);
  Value rt = createActivationByName(b, activations.f, r);

  Value h;
  if (linear_before_reset) {
    Value h_x = b.create<AtenLinearOp>(hTy, Xt, weights.Wi, weights.Wbi);
    Value h_r = b.create<AtenMulTensorOp>(hTy, rt, H_prev);
    Value h_h = b.create<AtenLinearOp>(hTy, h_r, weights.Ri, weights.Rbi);
    h = b.create<AtenAddTensorOp>(hTy, h_x, h_h, cstOne);
  } else {
    Value h_x = b.create<AtenLinearOp>(hTy, Xt, weights.Wi, weights.Wbi);
    Value h_r = b.create<AtenMulTensorOp>(hTy, rt, H_prev);
    Value h_h = b.create<AtenLinearOp>(hTy, h_r, weights.Ri, weights.Rbi);
    h = b.create<AtenAddTensorOp>(hTy, h_x, h_h, cstOne);
  }
  Value ht = createActivationByName(b, activations.g, h);

  Value zt_neg = b.create<AtenSubScalarOp>(hTy, cstOne, zt, cstOne);
  Value ht_zt_neg = b.create<AtenMulTensorOp>(hTy, ht, zt_neg);
  Value H_prev_zt = b.create<AtenMulTensorOp>(hTy, H_prev, zt);
  Value H_new = b.create<AtenAddTensorOp>(hTy, ht_zt_neg, H_prev_zt, cstOne);

  return H_new;
}

GruLayerOutput gru_layer(ImplicitLocOpBuilder &b, Value X, Value initial_h,
                         GruWeights weights, GruActivations activations,
                         bool linear_before_reset) {
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

  std::string direction;

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
  Value B;
  if (binder.tensorOperandAtIndex(B, 3)) {
    B = b.create<AtenZerosOp>(W.getType(), W);
  }

  llvm::SmallVector<std::string> activationsList;

  GruActivations activations;
  activations.f = "Sigmoid";
  activations.g = "Tanh";

  if (!binder.stringArrayAttr(activationsList, "activations") &&
      activationsList.size() > 0) {
    if (activationsList.size() == 2) {
      activations.f = activationsList[0];
      activations.g = activationsList[1];
    } else {
      return rewriter.notifyMatchFailure(
          binder.op, "Unsupported number of activation functions: " +
                         std::to_string(activationsList.size()) +
                         " are provided.");
    }
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
  if (hidden_size != wTy.getSizes()[1])
    return rewriter.notifyMatchFailure(
        binder.op, "hidden_size (" + std::to_string(hidden_size) +
                       ") does not match the second dimension of wTy (" +
                       std::to_string(wTy.getSizes()[1]) + ")");
  if (wTy.getSizes()[2] != input_size * 3)
    return rewriter.notifyMatchFailure(
        binder.op, "The third dimension of wTy (" +
                       std::to_string(wTy.getSizes()[2]) +
                       ") does not match 3 * input_size (" +
                       std::to_string(3 * input_size) + ")");

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

  auto intType = b.getType<IntType>();

  Value cstNumDirections =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(num_directions));
  Value cstBatchSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(batch_size));
  Value cstHiddenSize =
      b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value cstNone = b.create<ConstantNoneOp>();
  Value cstZero = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(0));

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
  if (num_directions != 1) {
    return rewriter.notifyMatchFailure(
        binder.op, "Unsupported num_directions. Only 1 is supported but " +
                       std::to_string(num_directions) + " is provided.");
  }
  Value cstOne = b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(1));
  GruWeights weights;
  weights.Wi = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, input_size}, wTy.getDtype()),
      W_forward, cstZero, cstZero, cstHiddenSize, cstOne);
  weights.Ri = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, hidden_size}, wTy.getDtype()),
      W_forward, cstHiddenSize, cstZero,
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(2))),
      cstOne);
  weights.Wbi = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstZero, cstZero, cstHiddenSize, cstOne);
  weights.Rbi = b.create<AtenSliceTensorOp>(
      b.getType<ValueTensorType>(llvm::SmallVector<int64_t>{hidden_size},
                                 wTy.getDtype()),
      B_forward, cstHiddenSize, cstZero,
      b.create<AtenMulIntOp>(
          cstHiddenSize,
          b.create<ConstantIntOp>(intType, b.getI64IntegerAttr(2))),
      cstOne);
  int64_t linear_before_reset_int;
  if (binder.s64IntegerAttr(linear_before_reset_int, "linear_before_reset", 0))
    return rewriter.notifyMatchFailure(
        binder.op,
        "Missing or invalid linear_before_reset attribute. Using default "
        "value of 0.");
  bool linear_before_reset = linear_before_reset_int != 0;
  GruLayerOutput gruLayerOutput = gru_layer(b, X, initial_h_forward, weights,
                                            activations, linear_before_reset);
  Value Y_unsqueezed = b.create<AtenUnsqueezeOp>(yTy, gruLayerOutput.Y, cstOne);
  auto Y_h_unsqueezed_type = b.getType<ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      cast<ValueTensorType>(gruLayerOutput.Y_h.getType()).getDtype());
  Value Y_h_unsqueezed = b.create<AtenUnsqueezeOp>(Y_h_unsqueezed_type,
                                                   gruLayerOutput.Y_h, cstZero);
  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
