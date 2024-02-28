#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

// debug
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "torch-onnx-to-torch-patterns"
namespace mlir::torch::onnx_c {

std::pair<Value, Value>
lstm_cell(OpBinder binder, ConversionPatternRewriter &rewriter,
          Value Xt, // =input_seq shape [batch_size, input_size]
          Value H,  // =hidden_state shape [batch_size, hidden_size]
          Value C,  // =cell_state shape [batch_size, hidden_size]
          Value W,  // =weights_hh shape [hidden_size*4, input_size]
          Value Wb, // =bias_hh shape [hidden_size*8]
          Value R,  // =weights_hr shape [hidden_size*4, hidden_size]
          Value Rb
          // Value P, // =peephole shape [hidden_size*3]; not supported yet
) {
  // this function is made based on
  // https://github.com/pytorch/pytorch/pull/91465
  // and the names should match
  // it returns the hidden state and cell state after the lstm cell
  // TODO: it is possible to combine the 4 matrix multiplications into a single
  // one
  //       by concatenating the weights and inputs into a single matrix
  // assume that num_directions is already eliminated
  // Python: hidden_size = Ht_prev.size(1)
  Torch::ValueTensorType XType = Xt.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType HType = H.getType().cast<Torch::ValueTensorType>();
  int64_t hidden_size = HType.getSizes()[1];
  int64_t batch_size = HType.getSizes()[0];

  // helper lambda function for getting type of tiled tensor
  auto getTiledType = [&](Torch::ValueTensorType input_type, int64_t tile_dim,
                          int64_t tile_factor) {
    return rewriter
        .getType<Torch::ValueTensorType>(
            llvm::SmallVector<int64_t>{input_type.getSizes()[0],
                                       4 * input_type.getSizes()[1]},
            input_type.getDtype())
        .cast<Torch::ValueTensorType>();
  };

  Torch::ValueTensorType XTiledType = getTiledType(XType, 1, 4);
  Torch::ValueTensorType HTiledType = getTiledType(HType, 1, 4);

  // some constant values
  Value c_0 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  Value c_1 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
  Value c_4 =
      rewriter.create<Torch::ConstantIntOp>( // for blowing up Xt and H to
                                             // multiply into W for the 4 gates
          binder.getLoc(), rewriter.getType<Torch::IntType>(),
          rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4));

  SmallVector<Value> dimList = {
      c_1, c_4}; // don't blow up batch, blow up input size by 4

  Value dimListValue = rewriter.create<Torch::PrimListConstructOp>(
      binder.getLoc(),
      rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
      dimList);

  // make AnyTorchListOfTorchIntType for Tile op. it should be [1,]

  // first tile input seq and hidden state by 4
  Value XTiled = rewriter.create<Torch::AtenTileOp>(
      /*loc*/ binder.getLoc(), /*type*/ XTiledType,
      /*self=*/Xt,
      /*dims=*/dimListValue);
  Value HTiled = rewriter.create<Torch::AtenTileOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*self=*/H,
      /*dims=*/dimListValue);

  //// Assemble gates
  // input term
  // Python: gates = torch.ops.aten.linear(Xt, W, Wb) +
  // torch.ops.aten.linear(H_prev, R, Rb)
  Value G_x = rewriter.create<Torch::AtenLinearOp>(
      /*loc*/ binder.getLoc(), /*type*/ XTiledType,
      /*input=*/XTiled, /*weight=*/W, /*bias=*/Wb);
  // hidden / recurrence term
  Value G_h = rewriter.create<Torch::AtenLinearOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*input=*/HTiled, /*weight=*/R, /*bias=*/Rb);
  Value G = rewriter.create<Torch::AtenAddTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*input=*/G_x, /*other=*/G_h, /*alpha=*/c_1);

  Value HSizeX1 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), hidden_size));

  Value HSizeX2 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 2 * hidden_size));

  Value HSizeX3 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 3 * hidden_size));

  Value HSizeX4 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4 * hidden_size));

  // Gate Vector splitting
  // Split G[iofc] into iof and c
  // Python:
  // sigmoid_chunk = gates[:, :3 * hidden_size]
  Value G_iof = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*input=*/G, /*dim=*/c_1, /*start=*/c_0, /*end=*/HSizeX3,
      /*step=*/c_1);
  // Python:
  // tanh_chunk = gates[:, 3 * hidden_size:]
  Value G_c = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*input=*/G, /*dim=*/c_1, /*start=*/HSizeX3, /*end=*/HSizeX4,
      /*step=*/c_1);

  // Activation for IOF
  // TODO: activations: support non-default activations
  Torch::ValueTensorType IOFType = rewriter.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{batch_size, hidden_size * 3},
      Xt.getType().cast<Torch::ValueTensorType>().getDtype());
  // sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
  Value Activation_IOF = rewriter.create<Torch::AtenSigmoidOp>(
      /*loc*/ binder.getLoc(), /*type*/ IOFType, /*input=*/G_iof);
  // Update_I = sigmoid_chunk[:, :hidden_size]

  // Further splitting
  Value Activation_I = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_IOF, /*dim=*/c_1, /*start=*/c_0, /*end=*/HSizeX1,
      /*step=*/c_1);
  Value Activation_O = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_IOF, /*dim=*/c_1, /*start=*/HSizeX1, /*end=*/HSizeX2,
      /*step=*/c_1);
  Value Activation_F = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_IOF, /*dim=*/c_1, /*start=*/HSizeX2, /*end=*/HSizeX3,
      /*step=*/c_1);
  // Activation for C
  // TODO: activations: support non-default activations
  // this is activation g in the ONNX docs
  Value Activation_C = rewriter.create<Torch::AtenTanhOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType, /*input=*/G_c); // HType

  // Ct = ft (.) Ct-1 + it (.) ct, where (.) is element-wise multiplication
  // Python:
  // Ct = (ft * Ct_prev) + (it * ct)
  Value C_forget_contribution = rewriter.create<Torch::AtenMulTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_F,
      /*other=*/C); // this is really the part of C that is NOT forgotten
  Value C_input_contribution = rewriter.create<Torch::AtenMulTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_I, /*other=*/Activation_C);
  Value C_new = rewriter.create<Torch::AtenAddTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/C_forget_contribution, /*other=*/C_input_contribution,
      /*alpha=*/c_1);

  // TODO: activations: replace tanh with op corresponding to activations[2]
  // Ht = ot (.) h(Ct)  where h is tanh by default
  // this is activation h in the ONNX docs
  Value C_new_tanh = rewriter.create<Torch::AtenTanhOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType, /*input=*/C_new);
  Value H_new = rewriter.create<Torch::AtenMulTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_O, /*other=*/C_new_tanh);

  return std::make_pair(H_new, C_new);
}

std::tuple<Value, Value, Value> lstm_layer( // returns Y, Y_h, Y_c
    OpBinder binder, ConversionPatternRewriter &rewriter,
    Value X,         // =input_seq shape [seq_length, batch_size, input_size]
    Value initial_h, // =hidden_state shape [num_layers*num_directions,
                     // batch_size, hidden_size]
    Value initial_c, // =cell_state shape [num_layers*num_directions,
                     // batch_size, hidden_size]
    Value W,  // =weights_hh shape [num_layers*num_directions, hidden_size*4,
              // input_size]
    Value Wb, // =bias_hh shape [num_layers*num_directions, hidden_size*8]
    Value R,  // =weights_hr shape [num_layers*num_directions, hidden_size*4,
              // hidden_size]
    Value Rb
    // Value P, // =peephole shape [num_layers*num_directions, hidden_size*3];
    // not supported yet
) {

  Location loc = binder.getLoc();
  int64_t seq_len = X.getType().cast<Torch::ValueTensorType>().getSizes()[0];
  int64_t batch_size = X.getType().cast<Torch::ValueTensorType>().getSizes()[1];
  int64_t input_size = X.getType().cast<Torch::ValueTensorType>().getSizes()[2];
  int64_t hidden_size =
      initial_h.getType().cast<Torch::ValueTensorType>().getSizes()[2];

  Torch::ValueTensorType HType =
      initial_h.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType CType =
      initial_c.getType()
          .cast<Torch::ValueTensorType>(); // should be same as HType

  // Y = history of hidden states
  Torch::ListType YType = rewriter.getType<Torch::ListType>(HType);

  Value Y =
      rewriter.create<Torch::PrimListConstructOp>(loc, YType, ValueRange({}));
  // Create a for-like PrimLoopOp.
  Value maxTripCount = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), seq_len));
  Value cTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);

  Value cstOne = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

  // iter args:
  //  i: loop index
  //  Y: history of hidden states
  //  H: hidden state
  //  C: cell state
  auto loop = rewriter.create<Torch::PrimLoopOp>(
      loc, /*results=*/TypeRange({cstOne.getType(), Y.getType(), HType, CType}),
      maxTripCount,
      /*initialCondition=*/cTrue,
      /*iterArgsInit=*/ValueRange({cstOne, Y, initial_h, initial_c}));

  // Create the loop body.
  {
    OpBuilder::InsertionGuard guard(rewriter);

    Block *body =
        rewriter.createBlock(/*parentRegion=*/&loop.getRegion(),
                             /*insertionPoint=*/loop.getRegion().begin(),
                             /*argumentTypes=*/
                             TypeRange({rewriter.getType<Torch::IntType>(),
                                        Y.getType(), HType, CType}),
                             /*locations=*/{loc, loc, loc, loc});

    Value iterationNumber = body->getArgument(0);
    Value history = body->getArgument(1);
    Value H_prev = body->getArgument(2);
    Value C_prev = body->getArgument(3);

    Torch::ValueTensorType XType = X.getType().cast<Torch::ValueTensorType>();
    Torch::ValueTensorType XtType = rewriter.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, XType.getDtype());

    Value cstTensorOne = rewriter.create<Torch::PrimNumToTensorScalarOp>(
        binder.getLoc(),
        Torch::ValueTensorType::get(
            rewriter.getContext(),
            /*shape*/ ArrayRef<int64_t>{1},
            /*dtype*/ rewriter.getIntegerType(64, /*signed*/ 1)),
        cstOne);

    Value Xt = rewriter.create<Torch::AtenIndexSelectOp>(
        loc, XtType, X, iterationNumber, cstTensorOne);

    auto [H_new, C_new] =
        lstm_cell(binder, rewriter, Xt, H_prev, C_prev, W, Wb, R, Rb);

    // append H_new to Y
    rewriter.create<Torch::AtenAppendTOp>(loc, Y.getType(), history, H_new);

    Value nextIterationNumber = rewriter.create<Torch::AtenAddIntOp>(
        loc, rewriter.getType<Torch::IntType>(), iterationNumber, cstOne);

    Value loopCondition = rewriter.create<Torch::AtenLtIntOp>(
        loc, rewriter.getType<Torch::BoolType>(), nextIterationNumber,
        maxTripCount);

    rewriter.create<Torch::PrimLoopConditionOp>(
        loc, /*shouldContinue=*/loopCondition,
        /*iterArgs=*/ValueRange({nextIterationNumber, Y, H_new, C_new}));
  }

  // Return the result of the loop.
  return std::make_tuple(Y, loop.getResult(2), loop.getResult(3));
}

LogicalResult OnnxLSTMHandler(OpBinder binder,
                              ConversionPatternRewriter &rewriter) {
  // TODO: maybe move LSTM Handler and its helpers to their own file

  auto loc = binder.getLoc();

  // required inputs
  Value X, W, R;
  Torch::ValueTensorType XType, WType, RType;
  // required attributes
  int64_t hidden_size;

  // optional inputs
  Value B, sequence_lens, initial_h, initial_c, P;
  Torch::ValueTensorType BType, sequence_lensType, initial_hType, initial_cType,
      PType;

  // optional attributes
  float activation_alpha, activation_beta;
  llvm::SmallVector<std::string> activations;
  float clip;
  std::string direction;
  int64_t input_forget;

  // result types
  Torch::ValueTensorType YType, Y_hType, Y_cType;

  if (!binder.tensorResultType(YType) && !binder.tensorResultType(Y_hType) &&
      !binder.tensorResultType(Y_cType)) {
    LLVM_DEBUG(llvm::dbgs()
               << "LSTM: At least one of the outputs must be present\n");
    return failure();
  }

  // fail if required attributes/inputs/outputs are not found
  if (binder.tensorOperandAtIndex(X, 0) || binder.tensorOperandAtIndex(W, 1) ||
      binder.tensorOperandAtIndex(R, 2) ||
      binder.s64IntegerAttr(hidden_size, "hidden_size")) {
    LLVM_DEBUG(llvm::dbgs()
               << "LSTM: Required inputs/attributes are not found\n");
    return failure();
  }

  XType = X.getType().cast<Torch::ValueTensorType>();
  WType = W.getType().cast<Torch::ValueTensorType>();
  RType = R.getType().cast<Torch::ValueTensorType>();

  if (!binder.customOpNameStringAttr(direction, "direction") &&
      direction != "forward" && direction != "") {
    LLVM_DEBUG(llvm::dbgs() << "LSTM: Only forward direction is supported\n");
    // when implementing bidirectional lstm:
    // when the reverse direction is present,
    // W will be the concatination of W and W_reverse along dim 0
    // (and same for R and B)
    return failure();
  }

  if (binder.tensorOperandAtIndex(B, 3)) {
    // set B to be a zero tensor if not provided
    BType =
        WType
            .getWithSizesAndDtype(llvm::SmallVector<int64_t>{hidden_size * 8},
                                  WType.getDtype())
            .cast<Torch::ValueTensorType>();
    B = rewriter.create<Torch::AtenZerosOp>(binder.getLoc(), W.getType(), W);
  }

  // we are currently only some of the optional inputs/outputs
  // fail if optional inputs/outputs are found
  if (!binder.tensorOperandAtIndex(sequence_lens, 4) ||
          !binder.tensorOperandAtIndex(P, 7) ||
          !binder.f32FloatAttr(activation_alpha, "activation_alpha", 1.0f) ||
          !binder.f32FloatAttr(activation_beta, "activation_beta", 0.0f) ||
          !binder.f32FloatAttr(clip, "clip", 0.0f) ||
          !binder.s64IntegerAttr(input_forget, "input_forget"),
      0) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "LSTM: Optional inputs/outputs/attributes are not supported\n");
    return failure();
  }

  // TODO: activations: edit this check when custom activations are supported
  if (!binder.stringArrayAttr(activations, "activations")) {
    if (activations.size() == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "LSTM: No activations are provided; using defaults\n");
      activations.push_back("Sigmoid");
      activations.push_back("Tanh");
      activations.push_back("Tanh");
    }
    if (activations.size() != 3) {
      LLVM_DEBUG(llvm::dbgs() << "LSTM: Only 3 activations are supported but "
                              << activations.size() << " are provided\n");
      return failure();
    }

    // activations found. make sure they are the default ones
    if (activations[0] != "Sigmoid" || activations[1] != "Tanh" ||
        activations[2] != "Tanh") {
      // Default: f=Sigmoid, g=Tanh, h=Tanh
      LLVM_DEBUG(llvm::dbgs()
                 << "LSTM: Only default activations are supported\n");
      return failure();
    }
  }

  if (!binder.customOpNameStringAttr(direction, "direction", "forward") &&
      direction != "forward") {

    LLVM_DEBUG(llvm::dbgs() << "LSTM: Only forward direction is supported\n");
    return failure();
  }
  int64_t num_directions = 1 + (direction == "bidirectional");

  // get W type
  // assert that hidden_size is consistent with HType.getsizes()[1]

  assert(num_directions == WType.getSizes()[0]);
  assert(num_directions == 1);
  assert(4 * hidden_size == WType.getSizes()[1]);

  auto XShape = XType.getSizes();
  int64_t seq_lengths = XShape[0]; // number of timesteps
  int64_t batch_size = XShape[1];
  int64_t input_size = XShape[2]; // number of features

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
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(
        binder.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
    Value cstDirection = rewriter.create<Torch::ConstantIntOp>(
        binder.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), direction));

    Value directionTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
        binder.getLoc(),
        Torch::ValueTensorType::get(
            rewriter.getContext(),
            /*shape*/ ArrayRef<int64_t>{1},
            /*dtype*/ rewriter.getIntegerType(64, /*signed*/ 1)),
        cstDirection);

    // why does rewriter.getType not work but Torch::ValueTensorType::get
    // work??? it claims that i64 is invalid dtype for torch !vtensor Value
    // directionTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
    //     binder.getLoc(),
    //     rewriter.getType<Torch::ValueTensorType>(
    //         /*shape*/ArrayRef<int64_t>{1},
    //         /*dtype*/rewriter.getIntegerType(64, /*signed*/ 1)),
    //     cstDirection);
    return rewriter.create<Torch::AtenIndexSelectOp>(
        binder.getLoc(), outputType, input, cstZero,
        /*AnyTorchTensorType:$index*/ directionTensor);
  };

  Value W_forward = getDirection(0, W);
  Value R_forward = getDirection(0, R);
  Value B_forward = getDirection(0, B);

  // ### everything hereon is only one direction ###

  Torch::ValueTensorType HType = rewriter.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{num_directions, batch_size, hidden_size},
      XType.getDtype());

  // construct a list containing the shape of initial_h and initial_c
  // this is used to check if initial_h and initial_c are provided
  Value cst_num_directions = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), num_directions));
  Value cst_batch_size = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), batch_size));
  Value cst_hidden_size = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), hidden_size));
  Value cst_None = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

  Value initial_hc_shape = rewriter.create<Torch::PrimListConstructOp>(
      binder.getLoc(),
      rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
      ValueRange({cst_num_directions, cst_batch_size, cst_hidden_size}));

  Value cstDtype = Torch::getDtypeIntValueForType(rewriter, binder.getLoc(),
                                                  XType.getDtype());

  // initialize hidden and cell states
  if (binder.tensorOperandAtIndex(initial_h, 5)) {
    LLVM_DEBUG(llvm::dbgs()
               << "LSTM: initial_h not found; initializing to zeros\n");
    initial_h = rewriter.create<Torch::AtenZerosOp>(binder.getLoc(), HType,
                                                    /*size*/ initial_hc_shape,
                                                    /*dtype*/ cstDtype,
                                                    /*layout*/ cst_None,
                                                    /*device*/ cst_None,
                                                    /*pin_memory*/ cst_None);
  }
  if (binder.tensorOperandAtIndex(initial_c, 6)) {
    LLVM_DEBUG(llvm::dbgs()
               << "LSTM: initial_c not found; initializing to zeros\n");
    initial_c = rewriter.create<Torch::AtenZerosOp>(binder.getLoc(), HType,
                                                    /*size*/ initial_hc_shape,
                                                    /*dtype*/ cstDtype,
                                                    /*layout*/ cst_None,
                                                    /*device*/ cst_None,
                                                    /*pin_memory*/ cst_None);
  }

  // run lstm layer
  auto [Y, Y_h, Y_c] = lstm_layer(binder, rewriter, X, initial_h, initial_c,
                                  W_forward, B_forward, R_forward, B_forward);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y, Y_h, Y_c});
  return success();
}
} // namespace mlir::torch::onnx_c