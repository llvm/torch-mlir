#pragma once

// **************frequently-used macro**************
// assert macro
#include <cstdio>
#define llvm_assert(exp, ...)                                                  \
  if (exp) {                                                                   \
    printf(__VA_ARGS__);                                                       \
    return;                                                                    \
  }
#define input_assert(exp, ...)                                                 \
  llvm_assert(exp, "input error, require: " __VA_ARGS__)
#define llvm_assert_ret(exp, ret, ...)                                         \
  if (exp) {                                                                   \
    printf(__VA_ARGS__);                                                       \
    return ret;                                                                \
  }
#define input_assert_ret(exp, ret, ...)                                        \
  llvm_assert_ret(exp, ret, "input error, require: " __VA_ARGS__)
// debug macro
#define print_line() printf("line = %d\n", __LINE__)
#define print_value(value) llvm::outs() << value << '\n'

// **************package frequently-used***************
// package with func and class
#include "PassDetail.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using std::vector;

// frequently-used function about getting ops
typedef llvm::SmallPtrSet<Operation *, 16> OpList;
bool getConvMiddleOps(OpList &oplist, Operation *f, int layer);
bool getConvOp(OpList &oplist, Operation *f, int layer);

// frequently-used function about tensor
inline vector<int64_t> getShape(Value tensorOp) {
  // kernel shape: out_channels, in_channels, height, width
  // bias shape: out_channels
  return tensorOp.getType().cast<ValueTensorType>().getSizes().vec();
}
inline void toStdShape(vector<int64_t> &shape) {
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
}
inline void toBiasShape(vector<int64_t> &kernelShape) {
  kernelShape.erase(kernelShape.begin() + 1, kernelShape.end());
}
inline int getChannelSize(vector<int64_t> kernelShape) {
  return kernelShape[1] * kernelShape[2] * kernelShape[3];
}
inline int getKernelSize(vector<int64_t> kernelShape) {
  return kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
}

inline void pushBackVec(vector<float> &ktensor, vector<float> source, int start,
                        int size) {
  ktensor.insert(ktensor.end(), source.begin() + start,
                 source.begin() + start + size);
}
inline void pushBackVec(vector<float> &ktensor, int start, int size) {
  pushBackVec(ktensor, ktensor, start, size);
}
void copyTensor(vector<float> &ktensor, ValueTensorLiteralOp tensor);
void creatOneTensor(vector<float> &ktensor, int64_t len);

// frequently-used function about convolution operations
class RewriteOp {
private:
  MLIRContext *context;
  IRRewriter rewriter;
  AtenConvolutionOp convOp;
  Location loc;
  Value oldKernelOp;
  Value oldBiasOp;

public:
  RewriteOp(MLIRContext *context, AtenConvolutionOp &convOp)
      : context(context), rewriter(context), convOp(convOp),
        loc(convOp.getLoc()) {
    rewriter.setInsertionPoint(convOp);
    oldKernelOp = convOp.getOperand(1);
    oldBiasOp = convOp.getOperand(2);
  }
  void setConvOp(AtenConvolutionOp &convOp) {
    this->convOp = convOp;
    oldKernelOp = convOp.getOperand(1);
    oldBiasOp = convOp.getOperand(2);
  }
  Value getInput() { return convOp.getOperand(0); }
  Value getKernel() { return oldKernelOp; }
  Value getBias() { return oldBiasOp; }
  ValueTensorLiteralOp getKernelTensor() {
    return oldKernelOp.getDefiningOp<ValueTensorLiteralOp>();
  }
  ValueTensorLiteralOp getBiasTensor() {
    return oldBiasOp.getDefiningOp<ValueTensorLiteralOp>();
  }
  vector<int64_t> getKernelShape() { return getShape(oldKernelOp); }
  vector<int64_t> getBiasShape() { return getShape(oldBiasOp); }
  ValueTensorType getValueTensorType(vector<int64_t> shape) {
    return ValueTensorType::get(context, llvm::ArrayRef(shape),
                                rewriter.getF32Type());
  }
  ValueTensorType getLeastValueTensorType() {
    return ValueTensorType::getWithLeastStaticInformation(context);
  }
  DenseElementsAttr getTensorDense(vector<int64_t> shape,
                                   vector<float> tensor) {
    return DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
        llvm::ArrayRef(tensor));
  }
  Value createTensorOp(vector<int64_t> shape, vector<float> tensor) {
    auto tensorType = getValueTensorType(shape);
    auto tensorDense = getTensorDense(shape, tensor);
    return rewriter.create<ValueTensorLiteralOp>(loc, tensorType, tensorDense);
  }
  Value createIntOp(int64_t value) {
    return rewriter.create<ConstantIntOp>(loc,
                                          rewriter.getI64IntegerAttr(value));
  }
  Value createFloatOp(double value) {
    return rewriter.create<ConstantFloatOp>(loc,
                                            rewriter.getF64FloatAttr(value));
    ;
  }
  Value createConvOp(Type result, Value inputOp, Value weightOp, Value biasOp,
                     Value groupOp) {
    return rewriter.create<AtenConvolutionOp>(
        loc, result, inputOp, weightOp, biasOp, convOp.getOperand(3),
        convOp.getOperand(4), convOp.getOperand(5), convOp.getOperand(6),
        convOp.getOperand(7), groupOp);
  }
  Value createConvOp(Type result, Value inputOp, Value weightOp, Value biasOp) {
    return createConvOp(result, inputOp, weightOp, biasOp,
                        convOp.getOperand(8));
  }
  Value createConvOp(Value inputOp, Value weightOp, Value biasOp,
                     Value groupOp) {
    return createConvOp(inputOp.getType(), inputOp, weightOp, biasOp, groupOp);
  }
  Value createConvOp(Value inputOp, Value weightOp, Value biasOp) {
    return createConvOp(inputOp, weightOp, biasOp, convOp.getOperand(8));
  }
  void replaceConvOp(Value newInputOp) {
    Value newConv =
        createConvOp(convOp.getType(), newInputOp, oldKernelOp, oldBiasOp);
    rewriter.replaceOp(convOp, newConv);
  }
  void replaceTensorOp(ValueTensorLiteralOp &oldTensor, vector<int64_t> shape,
                       vector<float> tensor) {
    auto tensorType = getValueTensorType(shape);
    auto tensorDense = getTensorDense(shape, tensor);
    rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldTensor, tensorType,
                                                      tensorDense);
  }
  Value createAddTensorOp(Type result, Value tensor1, Value tensor2,
                          Value alpha) {
    return rewriter.create<AtenAddTensorOp>(loc, result, tensor1, tensor2,
                                            alpha);
  }
  Value createSliceTensorOp(vector<int64_t> branchShape, Value input, Value dim,
                            Value start, Value end) {
    auto branchTensorType = getValueTensorType(branchShape);
    auto step = createIntOp(1);
    return rewriter.create<AtenSliceTensorOp>(loc, branchTensorType, input, dim,
                                              start, end, step);
  }
  Value creatTensorListOp(ValueTensorType tensorType, vector<Value> tensorVec) {
    return rewriter.create<PrimListConstructOp>(loc, ListType::get(tensorType),
                                                ValueRange(tensorVec));
  }
  Value creatCatTensorOp(vector<int64_t> resultShape, Value dim,
                         vector<Value> tensorVec) {
    auto vtensorType = getLeastValueTensorType();
    auto tensorList = creatTensorListOp(vtensorType, tensorVec);
    auto resultType = getValueTensorType(resultShape);
    return rewriter.create<AtenCatOp>(loc, resultType, tensorList, dim);
  }
};

// *********************macro for pass********************************
// handle param
#define type_param(type, param) type param
#define notype_param(type, param) param
#define this_param(type, param) this->param
#define init_param(type, param) this->param = param
// handle param list
#define handle_param(n, micro, ...) handle_param##n(micro, __VA_ARGS__)
#define handle_param1(micro, type1, param1) micro(type1, param1)
#define handle_param2(micro, type1, param1, type2, param2)                     \
  micro(type1, param1), micro(type2, param2)
// namespace, class, function
#define use_pass(name, n, ...)                                                 \
  namespace {                                                                  \
  class name##Pass : public name##Base<name##Pass> {                           \
  public:                                                                      \
    name##Pass() = default;                                                    \
    name##Pass(handle_param(n, type_param, __VA_ARGS__)) {                     \
      handle_param(n, init_param, __VA_ARGS__);                                \
    }                                                                          \
    void runOnOperation() override {                                           \
      MLIRContext *context = &getContext();                                    \
      auto f = getOperation();                                                 \
      name(context, f, handle_param(n, this_param, __VA_ARGS__));              \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  std::unique_ptr<OperationPass<func::FuncOp>>                                 \
      mlir::torch::Torch::create##name##Pass(                                  \
          handle_param(n, type_param, __VA_ARGS__)) {                          \
    return std::make_unique<name##Pass>(                                       \
        handle_param(n, notype_param, __VA_ARGS__));                           \
  }
