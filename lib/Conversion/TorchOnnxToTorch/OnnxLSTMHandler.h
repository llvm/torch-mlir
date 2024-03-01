/*

TODO list:

- slice before doing anything to avoid type complexity
  - fewer types
  - fuse matmuls and activations
  - support activations
  - slice outside loop
- put my functions into their own CPP file
- get rid of comments and use self-documentating variable names
- after fixing scf loop issue, numerical evaluation against onnxruntime
- support optional inputs and attributes
  - peephole
  - don't do sequence_lengths because we can infer it from the shape of X
  - activations
  - direction
  - clip?
  - input_forget?
  - support bidirectional lstm
  - reverse




*/

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

// debug
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "torch-onnx-to-torch-patterns"
namespace mlir::torch::onnx_c {

void printTensorShape(const char *name, const Value &tensor) {
  auto sizes = tensor.getType().cast<Torch::ValueTensorType>().getSizes();
  llvm::dbgs() << name << " shape: [";
  for (size_t i = 0; i < sizes.size(); ++i) {
    llvm::dbgs() << sizes[i];
    if (i != sizes.size() - 1) {
      llvm::dbgs() << ", ";
    }
  }
  llvm::dbgs() << "]\n";
}

std::pair<Value, Value>
lstm_cell(OpBinder binder, ConversionPatternRewriter &rewriter,
          Value Xt, // =input_seq shape [batch_size, input_size]
          Value H,  // =hidden_state shape [batch_size, hidden_size]
          Value C,  // =cell_state shape [batch_size, hidden_size]
          Value W,  // =weights_hh shape [hidden_size*4, input_size]
          Value Wb, // =bias_hh shape [hidden_size*4]
          Value R,  // =weights_hr shape [hidden_size*4, hidden_size]
          Value Rb  // =bias_hr shape [hidden_size*4]
          // Value P, // =peephole shape [hidden_size*3]; not supported yet
) {
  // this function is made based on
  // https://github.com/pytorch/pytorch/pull/91465
  // and the names should match
  // it returns the hidden state and cell state after the lstm cell
  Torch::ValueTensorType XType = Xt.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType HType = H.getType().cast<Torch::ValueTensorType>();
  int64_t hidden_size = HType.getSizes()[1];
  int64_t batch_size = HType.getSizes()[0];
  int64_t input_size = XType.getSizes()[1];
  // use some assertions to check the shapes
  // batch size
  if (Xt.getType().cast<Torch::ValueTensorType>().getSizes()[0] !=
          H.getType().cast<Torch::ValueTensorType>().getSizes()[0] ||
      Xt.getType().cast<Torch::ValueTensorType>().getSizes()[0] !=
          C.getType().cast<Torch::ValueTensorType>().getSizes()[0] ||
      Xt.getType().cast<Torch::ValueTensorType>().getSizes()[1] !=
          W.getType().cast<Torch::ValueTensorType>().getSizes()[1] ||
      H.getType().cast<Torch::ValueTensorType>().getSizes()[1] !=
          W.getType().cast<Torch::ValueTensorType>().getSizes()[0] / 4 ||
      H.getType().cast<Torch::ValueTensorType>().getSizes()[1] !=
          R.getType().cast<Torch::ValueTensorType>().getSizes()[1] ||
      H.getType().cast<Torch::ValueTensorType>().getSizes()[1] !=
          R.getType().cast<Torch::ValueTensorType>().getSizes()[0] / 4 ||
      H.getType().cast<Torch::ValueTensorType>().getSizes()[1] !=
          Wb.getType().cast<Torch::ValueTensorType>().getSizes()[0] / 4 ||
      H.getType().cast<Torch::ValueTensorType>().getSizes()[1] !=
          Rb.getType().cast<Torch::ValueTensorType>().getSizes()[0] / 4) {
    LLVM_DEBUG(llvm::dbgs() << "LSTM: input shapes are not consistent\n");
    LLVM_DEBUG(llvm::dbgs() << "Expected Xt shape:[" << batch_size << ","
                            << input_size << "]"
                            << "\n");
    printTensorShape("Xt", Xt);
    LLVM_DEBUG(llvm::dbgs() << "Expected H shape:[" << batch_size << ","
                            << hidden_size << "]"
                            << "\n");
    printTensorShape("H", H);
    LLVM_DEBUG(llvm::dbgs() << "Expected C shape:[" << batch_size << ","
                            << hidden_size << "]"
                            << "\n");
    printTensorShape("C", C);
    LLVM_DEBUG(llvm::dbgs() << "Expected W shape:[" << hidden_size * 4 << ","
                            << input_size << "]"
                            << "\n");
    printTensorShape("W", W);
    LLVM_DEBUG(llvm::dbgs() << "Expected R shape:[" << hidden_size * 4 << ","
                            << hidden_size << "]"
                            << "\n");
    printTensorShape("R", R);
    LLVM_DEBUG(llvm::dbgs() << "Expected Wb shape:[" << hidden_size * 4 << "]"
                            << "\n");
    printTensorShape("Wb", Wb);
    LLVM_DEBUG(llvm::dbgs() << "Expected Rb shape:[" << hidden_size * 4 << "]"
                            << "\n");
    printTensorShape("Rb", Rb);
  }

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

  Value c_0 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  Value c_1 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
  Value c_4 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4));

  Value dimListValue = rewriter.create<Torch::PrimListConstructOp>(
      binder.getLoc(),
      rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
      ArrayRef<Value>{c_1, c_4});

  Value XTiled = rewriter.create<Torch::AtenTileOp>(
      /*loc*/ binder.getLoc(), /*type*/ XTiledType,
      /*self=*/Xt,
      /*dims=*/dimListValue);
  Value HTiled = rewriter.create<Torch::AtenTileOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*self=*/H,
      /*dims=*/dimListValue);

  // gates = linear(Xt, W, Wb) + linear(H_prev, R, Rb)
  Value G_x = rewriter.create<Torch::AtenLinearOp>(
      /*loc*/ binder.getLoc(), /*type*/ HTiledType,
      /*input=*/XTiled, /*weight=*/W, /*bias=*/Wb);
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

  Torch::ValueTensorType G_iof_type = rewriter.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{batch_size, hidden_size * 3},
      Xt.getType().cast<Torch::ValueTensorType>().getDtype());

  // Gate Vector splitting
  // G_iof = gates[:, :3 * hidden_size]
  // G_c = gates[:, 3 * hidden_size:]
  Value G_iof = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ G_iof_type,
      /*input=*/G, /*dim=*/c_1, /*start=*/c_0, /*end=*/HSizeX3,
      /*step=*/c_1);
  Value G_c = rewriter.create<Torch::AtenSliceTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/G, /*dim=*/c_1, /*start=*/HSizeX3, /*end=*/HSizeX4,
      /*step=*/c_1);

  // Activation for IOF
  // TODO: activations: support non-default activations
  // this is activation f in the ONNX docs
  Value Activation_IOF = rewriter.create<Torch::AtenSigmoidOp>(
      /*loc*/ binder.getLoc(), /*type*/ G_iof_type, /*input=*/G_iof);
  // Activation for C
  // TODO: activations: support non-default activations
  // this is activation g in the ONNX docs
  Value Activation_C = rewriter.create<Torch::AtenTanhOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType, /*input=*/G_c); // HType

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
  // this is activation h in the ONNX docs
  // Ht = ot (.) h(Ct)  where h is tanh by default
  Value C_new_tanh = rewriter.create<Torch::AtenTanhOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType, /*input=*/C_new);
  Value H_new = rewriter.create<Torch::AtenMulTensorOp>(
      /*loc*/ binder.getLoc(), /*type*/ HType,
      /*input=*/Activation_O, /*other=*/C_new_tanh);

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
    Value Rb
    // Rb shape [hidden_size*4]
) {

  Location loc = binder.getLoc();
  int64_t seq_len = X.getType().cast<Torch::ValueTensorType>().getSizes()[0];
  int64_t batch_size = X.getType().cast<Torch::ValueTensorType>().getSizes()[1];
  int64_t input_size = X.getType().cast<Torch::ValueTensorType>().getSizes()[2];
  int64_t hidden_size =
      initial_h.getType().cast<Torch::ValueTensorType>().getSizes()[1];

  // check sizes
  assert(initial_h.getType().cast<Torch::ValueTensorType>().getSizes()[0] ==
         batch_size);
  assert(initial_h.getType().cast<Torch::ValueTensorType>().getSizes()[1] ==
         hidden_size);
  assert(initial_c.getType().cast<Torch::ValueTensorType>().getSizes()[0] ==
         batch_size);
  assert(initial_c.getType().cast<Torch::ValueTensorType>().getSizes()[1] ==
         hidden_size);
  assert(W.getType().cast<Torch::ValueTensorType>().getSizes()[0] ==
         hidden_size * 4);
  assert(W.getType().cast<Torch::ValueTensorType>().getSizes()[1] ==
         input_size);
  assert(Wb.getType().cast<Torch::ValueTensorType>().getSizes()[0] ==
         hidden_size * 4);
  assert(R.getType().cast<Torch::ValueTensorType>().getSizes()[0] ==
         hidden_size * 4);
  assert(R.getType().cast<Torch::ValueTensorType>().getSizes()[1] ==
         hidden_size);
  assert(Rb.getType().cast<Torch::ValueTensorType>().getSizes()[0] ==
         hidden_size * 4);

  Torch::ValueTensorType HType =
      initial_h.getType().cast<Torch::ValueTensorType>();
  Torch::ValueTensorType CType =
      initial_c.getType()
          .cast<Torch::ValueTensorType>(); // should be same as HType

  Torch::ListType Y_listType = rewriter.getType<Torch::ListType>(HType);

  Value Y_list = rewriter.create<Torch::PrimListConstructOp>(loc, Y_listType,
                                                             ValueRange({}));
  // Create a for-like PrimLoopOp.
  Value maxTripCount = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), seq_len));
  Value cTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);

  Value cstZero = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

  Type loopIndexType = rewriter.getType<Torch::IntType>();
  auto loop = rewriter.create<Torch::PrimLoopOp>(
      loc, /*results=*/TypeRange({HType, CType}), maxTripCount,
      /*initialCondition=*/cTrue,
      /*iterArgsInit=*/ValueRange({initial_h, initial_c}));
  // Create the loop body.
  {
    OpBuilder::InsertionGuard guard(rewriter);

    Block *body = rewriter.createBlock(
        /*parentRegion=*/&loop.getRegion(),
        /*insertionPoint=*/loop.getRegion().begin(),
        /*argumentTypes=*/
        TypeRange({
            loopIndexType, // loop condition
            HType,
            CType,
        }),
        /*locations=*/{loc, loc, loc});

    Value loopIndex = body->getArgument(0);
    Value H_prev = body->getArgument(1);
    Value C_prev = body->getArgument(2);

    Torch::ValueTensorType XType = X.getType().cast<Torch::ValueTensorType>();
    Torch::ValueTensorType XtType = rewriter.getType<Torch::ValueTensorType>(
        llvm::SmallVector<int64_t>{batch_size, input_size}, XType.getDtype());

    // Value loopIndexTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
    //     binder.getLoc(),
    //     Torch::ValueTensorType::get(
    //         rewriter.getContext(),
    //         /*shape*/ ArrayRef<int64_t>{1},
    //         /*dtype*/ rewriter.getIntegerType(64, /*signed*/ 1)),
    //     loopIndex);

    Value Xt = rewriter.create<Torch::AtenSelectIntOp>(loc, XtType, X, cstZero,
                                                       loopIndex);

    auto [H_new, C_new] =
        lstm_cell(binder, rewriter, Xt, H_prev, C_prev, W, Wb, R, Rb);

    // append H_new to Y
    rewriter.create<Torch::AtenAppendTOp>(loc, Y_list.getType(), Y_list, H_new);

    // we don't need this because we're using maxTripCount, right?
    // Value loopCondition = rewriter.create<Torch::AtenLtIntOp>(
    //     loc, rewriter.getType<Torch::BoolType>(), nextLoopIndex,
    //     maxTripCount);

    rewriter.create<Torch::PrimLoopConditionOp>(
        loc, /*shouldContinue=*/cTrue,
        /*iterArgs=*/ValueRange({H_new, C_new}));
  }

  // Return the result of the loop.
  return std::make_tuple(Y_list, loop.getResult(0), loop.getResult(1));
}

LogicalResult OnnxLSTMHandler(OpBinder binder,
                              ConversionPatternRewriter &rewriter) {
  // TODO: maybe move LSTM Handler and its helpers to their own file
  // required inputs
  Value X, W, R;
  Torch::ValueTensorType XType, WType, RType;
  // required attributes
  int64_t hidden_size;

  // optional inputs
  // we skip sequence_lengths because we infer it from the shape of X
  Value B, initial_h, initial_c;
  Torch::ValueTensorType BType;
  // Value P;

  // optional attributes
  // float activation_alpha, activation_beta;
  llvm::SmallVector<std::string> activations;
  std::string direction;
  // float clip;
  // int64_t input_forget;

  // result types
  Torch::ValueTensorType YType, Y_hType, Y_cType;

  if (binder.tensorResultTypeAtIndex(YType, 0) ||
      binder.tensorResultTypeAtIndex(Y_hType, 1) ||
      binder.tensorResultTypeAtIndex(Y_cType, 2)) {
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
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(
        binder.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
    Value cstDirection = rewriter.create<Torch::ConstantIntOp>(
        binder.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), direction));

    return rewriter.create<Torch::AtenSelectIntOp>(
        binder.getLoc(), outputType, input, cstZero, cstDirection);
  };

  Value W_forward = getDirection(0, W);
  Value R_forward = getDirection(0, R);
  Value B_forward = getDirection(0, B);

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
  Value cst_Zero = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  Value cst_One = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

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

  Value initial_h_forward = getDirection(0, initial_h);
  Value initial_c_forward = getDirection(0, initial_c);

  // ### everything hereon is only one direction. they won't have the direction
  // dimension ### todo: support bidirectional and reverse LSTM. it's possible
  // that it could be done by just reusing the lstm_layer function

  Value HSizeX4 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 4 * hidden_size));
  Value HSizeX8 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 8 * hidden_size));

  Torch::ValueTensorType WbRbType = rewriter.getType<Torch::ValueTensorType>(
      llvm::SmallVector<int64_t>{hidden_size * 4}, WType.getDtype());

  Value Wb = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), WbRbType,
      /*input=*/B_forward, /*dim=*/cst_Zero, /*start=*/cst_Zero,
      /*end=*/HSizeX4, /*step=*/cst_One);
  Value Rb = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), WbRbType,
      /*input=*/B_forward, /*dim=*/cst_Zero, /*start=*/HSizeX4,
      /*end=*/HSizeX8, /*step=*/cst_One);

  auto [Y_list, Y_h, Y_c] =
      lstm_layer(binder, rewriter, X, initial_h_forward, initial_c_forward,
                 W_forward, Wb, R_forward, Rb);

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
      binder.getLoc(), Y_h_Y_c_unsqueezed_type, Y_h, cst_Zero);
  Value Y_c_unsqueezed = rewriter.create<Torch::AtenUnsqueezeOp>(
      binder.getLoc(), Y_h_Y_c_unsqueezed_type, Y_c, cst_Zero);

  Torch::ValueTensorType Y_nonumdirections_type =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{seq_lengths, batch_size, hidden_size},
          YType.cast<Torch::ValueTensorType>().getDtype());

  Value Y_nonumdirections = rewriter.create<Torch::AtenStackOp>(
      binder.getLoc(), Y_nonumdirections_type, Y_list,
      /*dim*/ cst_Zero);

  // unsqueeze num_directions dim1 of Y
  // to create the onnx.LSTM output shape [seq_length, num_directions,
  // batch_size, hidden_size]
  Value Y_unsqueezed = rewriter.create<Torch::AtenUnsqueezeOp>(
      binder.getLoc(), YType, Y_nonumdirections, cst_One);

  rewriter.replaceOp(binder.op, mlir::ValueRange{Y_unsqueezed, Y_h_unsqueezed,
                                                 Y_c_unsqueezed});
  return success();
}
} // namespace mlir::torch::onnx_c