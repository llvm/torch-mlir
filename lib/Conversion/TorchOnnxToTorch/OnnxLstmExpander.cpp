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

struct CellState {
  Value H;
  Value C;
};
/*
This function represents a Long Short-Term Memory (LSTM) cell operation.

Parameters:
- ImplicitLocOpBuilder& b: A builder for constructing operations.
- Value Xt: The input sequence. It has a shape of [batch_size, input_size].
- Value H_prev: The previous hidden state. It has a shape of [batch_size,
hidden_size].
- Value C_prev: The previous cell state. It has a shape of [batch_size,
hidden_size].
- Value W_i, W_o, W_f, W_c: The weight matrices for input, output, forget, and
cell gates. Each has a shape of [hidden_size, input_size].
- Value Wb_i, Wb_o, Wb_f, Wb_c: The bias vectors for input, output, forget, and
cell gates. Each has a shape of [hidden_size].
- Value R_i, R_o, R_f, R_c: The recurrent weight matrices for input, output,
forget, and cell gates. Each has a shape of [hidden_size, hidden_size].
- Value Rb_i, Rb_o, Rb_f, Rb_c: The recurrent bias vectors for input, output,
forget, and cell gates. Each has a shape of [hidden_size].
- SmallVector<std::string> activations: A vector of activation functions to be
used in the LSTM cell. The function returns a CellState object representing the
state of the LSTM cell after the operation.
*/
CellState lstm_cell(ImplicitLocOpBuilder &b, Value Xt, Value H_prev,
                    Value C_prev, Value W_i, Value W_o, Value W_f, Value W_c,
                    Value Wb_i, Value Wb_o, Value Wb_f, Value Wb_c, Value R_i,
                    Value R_o, Value R_f, Value R_c, Value Rb_i, Value Rb_o,
                    Value Rb_f, Value Rb_c, ArrayRef<std::string> activations) {

  auto intType = b.getType<Torch::IntType>();
  Torch::ValueTensorType hTy = H_prev.getType().cast<Torch::ValueTensorType>();

  Value cstOne =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(1));

  // Apply linear/matmul for each gate separately
  // names are consistent with ONNX LSTM documentation
  Value i_x = b.create<Torch::AtenLinearOp>(hTy, Xt, W_i, Wb_i);
  Value i_h = b.create<Torch::AtenLinearOp>(hTy, H_prev, R_i, Rb_i);
  Value i = b.create<Torch::AtenAddTensorOp>(hTy, i_x, i_h, cstOne);
  Value i_act = createActivationByName(b, activations[0], i);

  Value o_x = b.create<Torch::AtenLinearOp>(hTy, Xt, W_o, Wb_o);
  Value o_h = b.create<Torch::AtenLinearOp>(hTy, H_prev, R_o, Rb_o);
  Value o = b.create<Torch::AtenAddTensorOp>(hTy, o_x, o_h, cstOne);
  Value o_act = createActivationByName(b, activations[0], o);

  Value f_x = b.create<Torch::AtenLinearOp>(hTy, Xt, W_f, Wb_f);
  Value f_h = b.create<Torch::AtenLinearOp>(hTy, H_prev, R_f, Rb_f);
  Value f = b.create<Torch::AtenAddTensorOp>(hTy, f_x, f_h, cstOne);
  Value f_act = createActivationByName(b, activations[0], f);

  Value ct_x = b.create<Torch::AtenLinearOp>(hTy, Xt, W_c, Wb_c);
  Value ct_h = b.create<Torch::AtenLinearOp>(hTy, H_prev, R_c, Rb_c);
  Value ct = b.create<Torch::AtenAddTensorOp>(hTy, ct_x, ct_h, cstOne);
  Value ct_act = createActivationByName(b, activations[1], ct);

  Value C_forget = b.create<Torch::AtenMulTensorOp>(hTy, f_act, C_prev);
  Value C_input = b.create<Torch::AtenMulTensorOp>(hTy, i_act, ct_act);

  CellState newCellState;
  newCellState.C =
      b.create<Torch::AtenAddTensorOp>(hTy, C_forget, C_input, cstOne);
  Value C_new_act = createActivationByName(b, activations[2], newCellState.C);
  newCellState.H = b.create<Torch::AtenMulTensorOp>(hTy, o_act, C_new_act);

  return newCellState;
}

std::tuple<Value, Value, Value> lstm_layer( // returns Y, Y_h, Y_c
    ImplicitLocOpBuilder &b, Value X,
    // X shape [seq_length, batch_size, input_size]
    Value initial_h,
    // =hidden_state shape [batch_size, hidden_size]
    Value initial_c,
    // initial_c shape [batch_size, hidden_size]
    Value W,
    // W shape [hidden_size*4, input_size]
    Value Wb,
    // Wb shape [hidden_size*4]
    Value R,
    // R shape [hidden_size*4, hidden_size]
    Value Rb,
    // Rb shape [hidden_size*4]
    SmallVector<std::string> activations) {

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
  Value HSizeX1 =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(hidden_size));
  Value HSizeX2 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(2 * hidden_size));
  Value HSizeX3 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(3 * hidden_size));
  Value HSizeX4 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(4 * hidden_size));

  Torch::ValueTensorType gateWeightsTypeIH = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, input_size},
      W.getType().cast<Torch::ValueTensorType>().getDtype());
  Value W_i = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeIH, W, cstZero,
                                                 cstZero, HSizeX1, cstOne);
  Value W_o = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeIH, W, cstZero,
                                                 HSizeX1, HSizeX2, cstOne);
  Value W_f = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeIH, W, cstZero,
                                                 HSizeX2, HSizeX3, cstOne);
  Value W_c = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeIH, W, cstZero,
                                                 HSizeX3, HSizeX4, cstOne);

  Torch::ValueTensorType gateWeightsTypeHH = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size, hidden_size},
      R.getType().cast<Torch::ValueTensorType>().getDtype());
  Value R_i = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeHH, R, cstZero,
                                                 cstZero, HSizeX1, cstOne);
  Value R_o = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeHH, R, cstZero,
                                                 HSizeX1, HSizeX2, cstOne);
  Value R_f = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeHH, R, cstZero,
                                                 HSizeX2, HSizeX3, cstOne);
  Value R_c = b.create<Torch::AtenSliceTensorOp>(gateWeightsTypeHH, R, cstZero,
                                                 HSizeX3, HSizeX4, cstOne);

  Torch::ValueTensorType gateBiasType = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size},
      Wb.getType().cast<Torch::ValueTensorType>().getDtype());
  Value Wb_i = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                  cstZero, HSizeX1, cstOne);
  Value Wb_o = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                  HSizeX1, HSizeX2, cstOne);
  Value Wb_f = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                  HSizeX2, HSizeX3, cstOne);
  Value Wb_c = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Wb, cstZero,
                                                  HSizeX3, HSizeX4, cstOne);
  Value Rb_i = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                  cstZero, HSizeX1, cstOne);
  Value Rb_o = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                  HSizeX1, HSizeX2, cstOne);
  Value Rb_f = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                  HSizeX2, HSizeX3, cstOne);
  Value Rb_c = b.create<Torch::AtenSliceTensorOp>(gateBiasType, Rb, cstZero,
                                                  HSizeX3, HSizeX4, cstOne);

  auto YType = b.getType<Torch::ValueTensorType>(
      SmallVector<int64_t>{seq_len, batch_size, hidden_size}, hTy.getDtype());

  auto YShapeList = b.create<Torch::PrimListConstructOp>(
      b.getType<Torch::ListType>(intType),
      ValueRange({cstSeqLen, cstBatchSize, cstHiddenSize}));

  int hDtypeInt = (int)getScalarTypeForType(hTy.getDtype());
  Value hDtypeIntVal =
      b.create<ConstantIntOp>(loc, b.getI64IntegerAttr(hDtypeInt));

  Value Y_initial = b.create<Torch::AtenZerosOp>(
      YType, YShapeList, hDtypeIntVal, cstNone, cstNone, cstNone);

  // Create a for-like PrimLoopOp.
  Value maxTripCount =
      b.create<Torch::ConstantIntOp>(intType, b.getI64IntegerAttr(seq_len));
  Value cTrue = b.create<Torch::ConstantBoolOp>(true);

  Type loopIndexType = intType;
  auto loop = b.create<Torch::PrimLoopOp>(
      /*results=*/TypeRange({YType, hTy, cTy}), maxTripCount,
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
            YType,
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

    auto [H_new, C_new] = lstm_cell(b, Xt, H_prev, C_prev, W_i, W_o, W_f, W_c,
                                    Wb_i, Wb_o, Wb_f, Wb_c, R_i, R_o, R_f, R_c,
                                    Rb_i, Rb_o, Rb_f, Rb_c, activations);

    Type hTyUnsqueezed = b.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{1, batch_size, hidden_size}, hTy.getDtype());
    Value H_new_unsqueezed =
        b.create<Torch::AtenUnsqueezeOp>(hTyUnsqueezed, H_new, cstZero);

    auto loopIndexPlusOne =
        b.create<Torch::AtenAddIntOp>(intType, loopIndex, cstOne);
    Value Y_new = b.create<Torch::AtenSliceScatterOp>(
        YType, Y_prev, H_new_unsqueezed, cstZero, loopIndex, loopIndexPlusOne,
        cstOne);

    b.create<Torch::PrimLoopConditionOp>(
        /*shouldContinue=*/cTrue,
        /*iterArgs=*/ValueRange({Y_new, H_new, C_new}));
  }

  Value Y = loop.getResult(0);
  Value Y_h = loop.getResult(1);
  Value Y_c = loop.getResult(2);

  return std::make_tuple(Y, Y_h, Y_c);
}

LogicalResult OnnxLstmExpander(OpBinder binder,
                               ConversionPatternRewriter &rewriter) {
  // For shapes and meanings of the inputs, see
  // https://onnx.ai/onnx/operators/onnx__LSTM.html
  Location loc = binder.getLoc();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);

  Torch::ValueTensorType xTy, wTy, rTy;
  // required attributes

  // optional inputs
  Value B, initial_h, initial_c;
  Torch::ValueTensorType bTy;
  llvm::SmallVector<std::string> activations;
  std::string direction;

  // result types
  Torch::ValueTensorType YType, Y_hType, Y_cType;

  if (binder.tensorResultTypeAtIndex(YType, 0) ||
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

  xTy = X.getType().cast<Torch::ValueTensorType>();
  wTy = W.getType().cast<Torch::ValueTensorType>();
  rTy = R.getType().cast<Torch::ValueTensorType>();

  if (binder.tensorOperandAtIndex(B, 3)) {
    bTy = wTy.getWithSizesAndDtype(llvm::SmallVector<int64_t>{hidden_size * 8},
                                   wTy.getDtype())
              .cast<Torch::ValueTensorType>();
    B = b.create<Torch::AtenZerosOp>(W.getType(), W);
  }

  if (!binder.stringArrayAttr(activations, "activations")) {
    if (activations.size() == 0) {
      activations.push_back("Sigmoid");
      activations.push_back("Tanh");
      activations.push_back("Tanh");
    }
    if (activations.size() != 3) {
      return rewriter.notifyMatchFailure(
          binder.op, "Only 3 activations are supported but " +
                         std::to_string(activations.size()) + " are provided.");
    }
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

  // initialize hidden and cell states
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    initial_h = b.create<Torch::AtenZerosOp>(hTy, HShape, cstDtype, cstNone,
                                             cstNone, cstNone);
  }
  if (binder.tensorOperandAtIndex(initial_c, 6)) {
    initial_c = b.create<Torch::AtenZerosOp>(hTy, HShape, cstDtype, cstNone,
                                             cstNone, cstNone);
  }

  Value initial_h_forward = getDirection(0, initial_h);
  Value initial_c_forward = getDirection(0, initial_c);

  // ### everything hereon is only one direction. they won't have the direction
  // dimension. todo: support bidirectional and reverse LSTM,
  // possibly by doing each direction separately and combining
  Value HSizeX4 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(4 * hidden_size));
  Value HSizeX8 = b.create<Torch::ConstantIntOp>(
      intType, b.getI64IntegerAttr(8 * hidden_size));

  Torch::ValueTensorType biasType = b.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size * 4}, wTy.getDtype());

  Value Wb = b.create<Torch::AtenSliceTensorOp>(
      biasType,
      /*input=*/B_forward, /*dim=*/cstZero, /*start=*/cstZero,
      /*end=*/HSizeX4, /*step=*/cstOne);
  Value Rb = b.create<Torch::AtenSliceTensorOp>(
      biasType,
      /*input=*/B_forward, /*dim=*/cstZero, /*start=*/HSizeX4,
      /*end=*/HSizeX8, /*step=*/cstOne);

  auto [Y_nonumdirections, Y_h, Y_c] =
      lstm_layer(b, X, initial_h_forward, initial_c_forward, W_forward, Wb,
                 R_forward, Rb, activations);

  // ### everything hereon has to have the direction dimension again ###
  // unsqueeze dim0 of Y_H and Y_c
  // Y_h = Y_h.unsqueeze(0)
  // Y_c = Y_c.unsqueeze(0)
  assert(num_directions == 1); // TODO: support bidirectional LSTM by doing both
                               // directions and replacing Unsqueeze with Stack
  Torch::ValueTensorType Y_h_Y_c_unsqueezed_type =
      b.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
          Y_h.getType().cast<Torch::ValueTensorType>().getDtype());
  Value Y_h_unsqueezed =
      b.create<Torch::AtenUnsqueezeOp>(Y_h_Y_c_unsqueezed_type, Y_h, cstZero);
  Value Y_c_unsqueezed =
      b.create<Torch::AtenUnsqueezeOp>(Y_h_Y_c_unsqueezed_type, Y_c, cstZero);

  // unsqueeze num_directions dim1 of Y
  // to create the onnx.LSTM output shape [seq_length, num_directions,
  // batch_size, hidden_size]
  Value Y_unsqueezed =
      b.create<Torch::AtenUnsqueezeOp>(YType, Y_nonumdirections, cstOne);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed,
                                                 Y_c_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
