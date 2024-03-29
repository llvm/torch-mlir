#include "torch-mlir/Conversion/TorchOnnxToTorch/OnnxLstmExpander.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir::torch::onnx_c {

Value createActivationByName(ConversionPatternRewriter &rewriter, Location loc,
                             StringRef name, Value input) {
  if (name == "Sigmoid") {
    return rewriter.create<Torch::AtenSigmoidOp>(loc, input.getType(), input);
  } else if (name == "Tanh") {
    return rewriter.create<Torch::AtenTanhOp>(loc, input.getType(), input);
  } else if (name == "Relu") {
    return rewriter.create<Torch::AtenReluOp>(loc, input.getType(), input);
  } else {
    llvm_unreachable("Unsupported activation function");
  }
}

std::pair<Value, Value>
lstm_cell(OpBinder binder, ConversionPatternRewriter &rewriter,
          Value Xt,     // =input_seq shape [batch_size, input_size]
          Value H_prev, // =hidden_state shape [batch_size, hidden_size]
          Value C_prev, // =cell_state shape [batch_size, hidden_size]
          Value W_i, Value W_o, Value W_f, Value W_c,
          // W shape [hidden_size, input_size]
          Value Wb_i, Value Wb_o, Value Wb_f, Value Wb_c,
          // Wb shape [hidden_size]
          Value R_i, Value R_o, Value R_f, Value R_c,
          // R shape [hidden_size, hidden_size]
          Value Rb_i, Value Rb_o, Value Rb_f, Value Rb_c,
          // Rb shape [hidden_size]
          SmallVector<std::string> activations) {
  Location loc = binder.getLoc();
  auto intType = rewriter.getType<Torch::IntType>();
  Torch::ValueTensorType HType =
      H_prev.getType().cast<Torch::ValueTensorType>();

  Value cstOne = rewriter.create<Torch::ConstantIntOp>(
      loc, intType, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

  // Apply linear/matmul for each gate separately
  // names are consistent with ONNX LSTM documentation
  Value i_x = rewriter.create<Torch::AtenLinearOp>(loc, HType, Xt, W_i, Wb_i);
  Value i_h =
      rewriter.create<Torch::AtenLinearOp>(loc, HType, H_prev, R_i, Rb_i);
  Value i =
      rewriter.create<Torch::AtenAddTensorOp>(loc, HType, i_x, i_h, cstOne);
  Value i_act = createActivationByName(rewriter, loc, activations[0], i);

  Value o_x = rewriter.create<Torch::AtenLinearOp>(loc, HType, Xt, W_o, Wb_o);
  Value o_h =
      rewriter.create<Torch::AtenLinearOp>(loc, HType, H_prev, R_o, Rb_o);
  Value o =
      rewriter.create<Torch::AtenAddTensorOp>(loc, HType, o_x, o_h, cstOne);
  Value o_act = createActivationByName(rewriter, loc, activations[0], o);

  Value f_x = rewriter.create<Torch::AtenLinearOp>(loc, HType, Xt, W_f, Wb_f);
  Value f_h =
      rewriter.create<Torch::AtenLinearOp>(loc, HType, H_prev, R_f, Rb_f);
  Value f =
      rewriter.create<Torch::AtenAddTensorOp>(loc, HType, f_x, f_h, cstOne);
  Value f_act = createActivationByName(rewriter, loc, activations[0], f);

  Value ct_x = rewriter.create<Torch::AtenLinearOp>(loc, HType, Xt, W_c, Wb_c);
  Value ct_h =
      rewriter.create<Torch::AtenLinearOp>(loc, HType, H_prev, R_c, Rb_c);
  Value ct =
      rewriter.create<Torch::AtenAddTensorOp>(loc, HType, ct_x, ct_h, cstOne);
  Value ct_act = createActivationByName(rewriter, loc, activations[1], ct);

  Value C_forget =
      rewriter.create<Torch::AtenMulTensorOp>(loc, HType, f_act, C_prev);
  Value C_input =
      rewriter.create<Torch::AtenMulTensorOp>(loc, HType, i_act, ct_act);
  Value C_new = rewriter.create<Torch::AtenAddTensorOp>(loc, HType, C_forget,
                                                        C_input, cstOne);

  Value C_new_act =
      createActivationByName(rewriter, loc, activations[2], C_new);
  Value H_new =
      rewriter.create<Torch::AtenMulTensorOp>(loc, HType, o_act, C_new_act);

  return std::make_pair(H_new, C_new);
}

std::tuple<Value, Value, Value> lstm_layer( // returns Y, Y_h, Y_c
    OpBinder binder, ConversionPatternRewriter &rewriter, Value X,
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

  Location loc = binder.getLoc();
  // these names are snake_case for consistency with onnx.LSTM documentation
  int64_t seq_len = X.getType().cast<Torch::ValueTensorType>().getSizes()[0];
  int64_t batch_size = X.getType().cast<Torch::ValueTensorType>().getSizes()[1];
  int64_t input_size = X.getType().cast<Torch::ValueTensorType>().getSizes()[2];
  int64_t hidden_size =
      initial_h.getType().cast<Torch::ValueTensorType>().getSizes()[1];

  Torch::ValueTensorType HType =
      initial_h.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType CType = HType;

  auto intType = rewriter.getType<Torch::IntType>();

  Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
  Value cstZero = rewriter.create<Torch::ConstantIntOp>(
      loc, intType, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  Value cstOne = rewriter.create<Torch::ConstantIntOp>(
      loc, intType, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
  Value cstSeqLen = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), seq_len));
  Value cstBatchSize = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), batch_size));
  Value cstHiddenSize = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), hidden_size));
  Value HSizeX1 = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), hidden_size));
  Value HSizeX2 = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 2 * hidden_size));
  Value HSizeX3 = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 3 * hidden_size));
  Value HSizeX4 = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4 * hidden_size));

  Torch::ValueTensorType GateWeightsTypeIH =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, input_size},
          W.getType().cast<Torch::ValueTensorType>().getDtype());
  Value W_i = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeIH, W, cstZero, cstZero, HSizeX1, cstOne);
  Value W_o = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeIH, W, cstZero, HSizeX1, HSizeX2, cstOne);
  Value W_f = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeIH, W, cstZero, HSizeX2, HSizeX3, cstOne);
  Value W_c = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeIH, W, cstZero, HSizeX3, HSizeX4, cstOne);

  Torch::ValueTensorType GateWeightsTypeHH =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, hidden_size},
          R.getType().cast<Torch::ValueTensorType>().getDtype());
  Value R_i = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeHH, R, cstZero, cstZero, HSizeX1, cstOne);
  Value R_o = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeHH, R, cstZero, HSizeX1, HSizeX2, cstOne);
  Value R_f = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeHH, R, cstZero, HSizeX2, HSizeX3, cstOne);
  Value R_c = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateWeightsTypeHH, R, cstZero, HSizeX3, HSizeX4, cstOne);

  Torch::ValueTensorType GateBiasType =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size},
          Wb.getType().cast<Torch::ValueTensorType>().getDtype());
  Value Wb_i = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Wb, cstZero, cstZero, HSizeX1, cstOne);
  Value Wb_o = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Wb, cstZero, HSizeX1, HSizeX2, cstOne);
  Value Wb_f = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Wb, cstZero, HSizeX2, HSizeX3, cstOne);
  Value Wb_c = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Wb, cstZero, HSizeX3, HSizeX4, cstOne);
  Value Rb_i = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Rb, cstZero, cstZero, HSizeX1, cstOne);
  Value Rb_o = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Rb, cstZero, HSizeX1, HSizeX2, cstOne);
  Value Rb_f = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Rb, cstZero, HSizeX2, HSizeX3, cstOne);
  Value Rb_c = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, GateBiasType, Rb, cstZero, HSizeX3, HSizeX4, cstOne);

  auto YType = rewriter.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{seq_len, batch_size, hidden_size}, HType.getDtype());

  auto YShapeList = rewriter.create<Torch::PrimListConstructOp>(
      loc, rewriter.getType<Torch::ListType>(intType),
      ValueRange({cstSeqLen, cstBatchSize, cstHiddenSize}));

  Value Y_initial = rewriter.create<Torch::AtenZerosOp>(
      loc, YType, YShapeList,
      Torch::getDtypeIntValueForType(rewriter, loc, HType.getDtype()), cstNone,
      cstNone, cstNone);

  // Create a for-like PrimLoopOp.
  Value maxTripCount = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), seq_len));
  Value cTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);

  Type loopIndexType = intType;
  auto loop = rewriter.create<Torch::PrimLoopOp>(
      loc, /*results=*/TypeRange({YType, HType, CType}), maxTripCount,
      /*initialCondition=*/cTrue,
      /*iterArgsInit=*/ValueRange({Y_initial, initial_h, initial_c}));
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *loopBody = rewriter.createBlock(
        /*parentRegion=*/&loop.getRegion(),
        /*insertionPoint=*/loop.getRegion().begin(),
        /*argumentTypes=*/
        TypeRange({
            loopIndexType, // loop condition
            YType,
            HType,
            CType,
        }),
        /*locations=*/{loc, loc, loc, loc});

    Value loopIndex = loopBody->getArgument(0);
    Value Y_prev = loopBody->getArgument(1);
    Value H_prev = loopBody->getArgument(2);
    Value C_prev = loopBody->getArgument(3);

    Torch::ValueTensorType XType = X.getType().cast<Torch::ValueTensorType>();
    Torch::ValueTensorType XtType = rewriter.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, XType.getDtype());

    Value Xt = rewriter.create<Torch::AtenSelectIntOp>(loc, XtType, X, cstZero,
                                                       loopIndex);

    auto [H_new, C_new] = lstm_cell(
        binder, rewriter, Xt, H_prev, C_prev, W_i, W_o, W_f, W_c, Wb_i, Wb_o,
        Wb_f, Wb_c, R_i, R_o, R_f, R_c, Rb_i, Rb_o, Rb_f, Rb_c, activations);

    Type HTypeUnsqueezed = rewriter.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{1, batch_size, hidden_size},
        HType.getDtype());
    Value H_new_unsqueezed = rewriter.create<Torch::AtenUnsqueezeOp>(
        loc, HTypeUnsqueezed, H_new, cstZero);

    auto loopIndexPlusOne =
        rewriter.create<Torch::AtenAddIntOp>(loc, intType, loopIndex, cstOne);
    Value Y_new = rewriter.create<Torch::AtenSliceScatterOp>(
        loc, YType, Y_prev, H_new_unsqueezed, cstZero, loopIndex,
        loopIndexPlusOne, cstOne);

    rewriter.create<Torch::PrimLoopConditionOp>(
        loc,
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
  // required inputs
  Value X, W, R;
  Torch::ValueTensorType XType, WType, RType;
  // required attributes
  int64_t hidden_size;

  // optional inputs
  Value B, initial_h, initial_c;
  Torch::ValueTensorType BType;
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

  if (binder.tensorOperandAtIndex(X, 0))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor X");
  if (binder.tensorOperandAtIndex(W, 1))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor W");
  if (binder.tensorOperandAtIndex(R, 2))
    return rewriter.notifyMatchFailure(binder.op,
                                       "Missing required input tensor R");
  if (binder.s64IntegerAttr(hidden_size, "hidden_size"))
    return rewriter.notifyMatchFailure(
        binder.op, "Missing required attribute hidden_size");

  XType = X.getType().cast<Torch::ValueTensorType>();
  WType = W.getType().cast<Torch::ValueTensorType>();
  RType = R.getType().cast<Torch::ValueTensorType>();

  if (binder.tensorOperandAtIndex(B, 3)) {
    BType =
        WType
            .getWithSizesAndDtype(llvm::SmallVector<int64_t>{hidden_size * 8},
                                  WType.getDtype())
            .cast<Torch::ValueTensorType>();
    B = rewriter.create<Torch::AtenZerosOp>(loc, W.getType(), W);
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

  assert(num_directions == WType.getSizes()[0]);
  assert(num_directions == 1);
  assert(4 * hidden_size == WType.getSizes()[1]);

  auto XShape = XType.getSizes();
  int64_t batch_size = XShape[1];
  int64_t input_size = XShape[2];
  assert(WType.getSizes()[2] == input_size);

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
    auto intType = rewriter.getType<Torch::IntType>();
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(
        loc, intType, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
    Value cstDirection = rewriter.create<Torch::ConstantIntOp>(
        loc, intType,
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), direction));

    return rewriter.create<Torch::AtenSelectIntOp>(loc, outputType, input,
                                                   cstZero, cstDirection);
  };

  Value W_forward = getDirection(0, W);
  Value R_forward = getDirection(0, R);
  Value B_forward = getDirection(0, B);

  Torch::ValueTensorType HType = rewriter.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      XType.getDtype());

  auto intType = rewriter.getType<Torch::IntType>();

  // construct a list containing the shape of initial_h and initial_c
  // this is used to check if initial_h and initial_c are provided
  Value cstNumDirections = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), num_directions));
  Value cstBatchSize = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), batch_size));
  Value cstHiddenSize = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), hidden_size));
  Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
  Value cstZero = rewriter.create<Torch::ConstantIntOp>(
      loc, intType, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  Value cstOne = rewriter.create<Torch::ConstantIntOp>(
      loc, intType, rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

  Value HShape = rewriter.create<Torch::PrimListConstructOp>(
      loc, rewriter.getType<Torch::ListType>(intType),
      ValueRange({cstNumDirections, cstBatchSize, cstHiddenSize}));

  Value cstDtype =
      Torch::getDtypeIntValueForType(rewriter, loc, XType.getDtype());

  // initialize hidden and cell states
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    initial_h = rewriter.create<Torch::AtenZerosOp>(
        loc, HType, HShape, cstDtype, cstNone, cstNone, cstNone);
  }
  if (binder.tensorOperandAtIndex(initial_c, 6)) {
    initial_c = rewriter.create<Torch::AtenZerosOp>(
        loc, HType, HShape, cstDtype, cstNone, cstNone, cstNone);
  }

  Value initial_h_forward = getDirection(0, initial_h);
  Value initial_c_forward = getDirection(0, initial_c);

  // ### everything hereon is only one direction. they won't have the direction
  // dimension. todo: support bidirectional and reverse LSTM,
  // possibly by doing each direction separately and combining
  Value HSizeX4 = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4 * hidden_size));
  Value HSizeX8 = rewriter.create<Torch::ConstantIntOp>(
      loc, intType,
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 8 * hidden_size));

  Torch::ValueTensorType biasType = rewriter.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size * 4}, WType.getDtype());

  Value Wb = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, biasType,
      /*input=*/B_forward, /*dim=*/cstZero, /*start=*/cstZero,
      /*end=*/HSizeX4, /*step=*/cstOne);
  Value Rb = rewriter.create<Torch::AtenSliceTensorOp>(
      loc, biasType,
      /*input=*/B_forward, /*dim=*/cstZero, /*start=*/HSizeX4,
      /*end=*/HSizeX8, /*step=*/cstOne);

  auto [Y_nonumdirections, Y_h, Y_c] =
      lstm_layer(binder, rewriter, X, initial_h_forward, initial_c_forward,
                 W_forward, Wb, R_forward, Rb, activations);

  // ### everything hereon has to have the direction dimension again ###
  // unsqueeze dim0 of Y_H and Y_c
  // Y_h = Y_h.unsqueeze(0)
  // Y_c = Y_c.unsqueeze(0)
  assert(num_directions == 1); // TODO: support bidirectional LSTM by doing both
                               // directions and replacing Unsqueeze with Stack
  Torch::ValueTensorType Y_h_Y_c_unsqueezed_type =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
          Y_h.getType().cast<Torch::ValueTensorType>().getDtype());
  Value Y_h_unsqueezed = rewriter.create<Torch::AtenUnsqueezeOp>(
      loc, Y_h_Y_c_unsqueezed_type, Y_h, cstZero);
  Value Y_c_unsqueezed = rewriter.create<Torch::AtenUnsqueezeOp>(
      loc, Y_h_Y_c_unsqueezed_type, Y_c, cstZero);

  // unsqueeze num_directions dim1 of Y
  // to create the onnx.LSTM output shape [seq_length, num_directions,
  // batch_size, hidden_size]
  Value Y_unsqueezed = rewriter.create<Torch::AtenUnsqueezeOp>(
      loc, YType, Y_nonumdirections, cstOne);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed,
                                                 Y_c_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c
