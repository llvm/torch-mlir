/*

TODO list:

- slice before doing anything to avoid type complexity [done]
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
          Value Xt,     // =input_seq shape [batch_size, input_size]
          Value H_prev, // =hidden_state shape [batch_size, hidden_size]
          Value C,      // =cell_state shape [batch_size, hidden_size]
          Value W_i, Value W_o, Value W_f, Value W_c,
          // W shape [hidden_size, input_size]
          Value Wb_i, Value Wb_o, Value Wb_f, Value Wb_c,
          // Wb shape [hidden_size]
          Value R_i, Value R_o, Value R_f, Value R_c,
          // R shape [hidden_size, hidden_size]
          Value Rb_i, Value Rb_o, Value Rb_f, Value Rb_c) {
  Torch::ValueTensorType HType =
      H_prev.getType().cast<Torch::ValueTensorType>();

  Value c_1 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

  // Apply linear/matmul for each gate separately
  Value I_x = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType, Xt,
                                                   W_i, Wb_i);
  Value I_h = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType,
                                                   H_prev, R_i, Rb_i);
  Value I = rewriter.create<Torch::AtenAddTensorOp>(binder.getLoc(), HType, I_x,
                                                    I_h, c_1);
  Value I_act =
      rewriter.create<Torch::AtenSigmoidOp>(binder.getLoc(), HType, I);

  Value O_x = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType, Xt,
                                                   W_o, Wb_o);
  Value O_h = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType,
                                                   H_prev, R_o, Rb_o);
  Value O = rewriter.create<Torch::AtenAddTensorOp>(binder.getLoc(), HType, O_x,
                                                    O_h, c_1);
  Value O_act =
      rewriter.create<Torch::AtenSigmoidOp>(binder.getLoc(), HType, O);

  Value F_x = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType, Xt,
                                                   W_f, Wb_f);
  Value F_h = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType,
                                                   H_prev, R_f, Rb_f);
  Value F = rewriter.create<Torch::AtenAddTensorOp>(binder.getLoc(), HType, F_x,
                                                    F_h, c_1);
  Value F_act =
      rewriter.create<Torch::AtenSigmoidOp>(binder.getLoc(), HType, F);

  Value C_x = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType, Xt,
                                                   W_c, Wb_c);
  Value C_h = rewriter.create<Torch::AtenLinearOp>(binder.getLoc(), HType,
                                                   H_prev, R_c, Rb_c);
  Value C_pre = rewriter.create<Torch::AtenAddTensorOp>(binder.getLoc(), HType,
                                                        C_x, C_h, c_1);
  Value C_act =
      rewriter.create<Torch::AtenTanhOp>(binder.getLoc(), HType, C_pre);

  Value C_forget =
      rewriter.create<Torch::AtenMulTensorOp>(binder.getLoc(), HType, F_act, C);
  Value C_input = rewriter.create<Torch::AtenMulTensorOp>(binder.getLoc(),
                                                          HType, I_act, C_act);
  Value C_new = rewriter.create<Torch::AtenAddTensorOp>(binder.getLoc(), HType,
                                                        C_forget, C_input, c_1);

  Value C_new_act =
      rewriter.create<Torch::AtenTanhOp>(binder.getLoc(), HType, C_new);
  Value H_new = rewriter.create<Torch::AtenMulTensorOp>(binder.getLoc(), HType,
                                                        O_act, C_new_act);

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

  Value c_0 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
  Value c_1 = rewriter.create<Torch::ConstantIntOp>(
      binder.getLoc(), rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
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

  Torch::ValueTensorType GateWeightsType =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size, input_size},
          W.getType().cast<Torch::ValueTensorType>().getDtype());

  Torch::ValueTensorType GateBiasType =
      rewriter.getType<Torch::ValueTensorType>(
          llvm::SmallVector<int64_t>{hidden_size},
          Wb.getType().cast<Torch::ValueTensorType>().getDtype());

  // Slice weights and biases for each gate
  Value W_i = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, W, c_0, c_0, HSizeX1, c_1);
  Value W_o = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, W, c_0, HSizeX1, HSizeX2, c_1);
  Value W_f = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, W, c_0, HSizeX2, HSizeX3, c_1);
  Value W_c = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, W, c_0, HSizeX3, HSizeX4, c_1);

  Value R_i = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, R, c_0, c_0, HSizeX1, c_1);
  Value R_o = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, R, c_0, HSizeX1, HSizeX2, c_1);
  Value R_f = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, R, c_0, HSizeX2, HSizeX3, c_1);
  Value R_c = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateWeightsType, R, c_0, HSizeX3, HSizeX4, c_1);

  Value Wb_i = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Wb, c_0, c_0, HSizeX1, c_1);
  Value Wb_o = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Wb, c_0, HSizeX1, HSizeX2, c_1);
  Value Wb_f = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Wb, c_0, HSizeX2, HSizeX3, c_1);
  Value Wb_c = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Wb, c_0, HSizeX3, HSizeX4, c_1);

  Value Rb_i = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Rb, c_0, c_0, HSizeX1, c_1);
  Value Rb_o = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Rb, c_0, HSizeX1, HSizeX2, c_1);
  Value Rb_f = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Rb, c_0, HSizeX2, HSizeX3, c_1);
  Value Rb_c = rewriter.create<Torch::AtenSliceTensorOp>(
      binder.getLoc(), GateBiasType, Rb, c_0, HSizeX3, HSizeX4, c_1);
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

    auto [H_new, C_new] = lstm_cell(binder, rewriter, Xt, H_prev, C_prev, W_i,
                                    W_o, W_f, W_c, Wb_i, Wb_o, Wb_f, Wb_c, R_i,
                                    R_o, R_f, R_c, Rb_i, Rb_o, Rb_f, Rb_c);

    rewriter.create<Torch::AtenAppendTOp>(loc, Y_list.getType(), Y_list, H_new);

    rewriter.create<Torch::PrimLoopConditionOp>(
        loc, /*shouldContinue=*/cTrue,
        /*iterArgs=*/ValueRange({H_new, C_new}));
  }

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