//assert macro
#include <cstdio>
#define llvm_assert(exp, ...) if(exp){ printf(__VA_ARGS__); return; }
#define input_assert(exp, ...) llvm_assert(exp, "input error, require: " __VA_ARGS__)

#define debug_assert()  printf("line = %d\n", __LINE__)
#define debug_assert1(value)  llvm::outs() << value << '\n'

// walk lambda
#define getMiddleOps(opWorklist, layer) [&](Operation *op) {    \
    if (isa<AtenConvolutionOp>(op)) {                           \
      layer--;                                      \
      if(layer == -1)                               \
        opWorklist.insert(op);                      \
    }                                               \
    if (layer == 0)                                 \
      opWorklist.insert(op);                        \
  } 

#define getConvOp(opWorklist, layer) [&](Operation *op) {  \
    if (isa<AtenConvolutionOp>(op)) {                      \
      layer--;                                      \
      if(layer == 0)                                \
        opWorklist.insert(op);                      \
    }                                               \
  } 

//frequently-used exp
#define getShape()   getType().cast<ValueTensorType>().getSizes().vec()
#define getBiasShape(shape)  shape.erase(shape.begin() + 1, shape.end())

#define getChannelSize(shape)  shape[1]*shape[2]*shape[3]
#define getKernelSize(shape)  shape[0]*shape[1]*shape[2]*shape[3]

#define getTensorType(context, shape, rewriter)   \
            ValueTensorType::get(context, llvm::ArrayRef(shape), rewriter.getF32Type())

#define getDense(shape, rewriter, vec)   DenseElementsAttr::get(        \
                    RankedTensorType::get(llvm::ArrayRef(shape),        \
                    rewriter.getF32Type()), llvm::ArrayRef(vec))

#define createIntOp(rewriter, loc, value)   \
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(value))

#define createTensorOp(rewriter, loc, tensorType, dense)  \
            rewriter.create<ValueTensorLiteralOp>(loc, tensorType, dense)

#define convParam_3to7(convOp)  convOp.getOperand(3), convOp.getOperand(4), \
                                convOp.getOperand(5), convOp.getOperand(6), \
                                convOp.getOperand(7)   

#define convParam_3to8(convOp)  convParam_3to7(convOp), convOp.getOperand(8)

//other exp
#define pushBackVec(vec1, vec2, start, size)    \
          vec1.insert(vec1.end(), vec2.begin() + start, vec2.begin() + start + size)

#define getTensor(tensorOp)  tensorOp.getValue().getValues<float>()

#define copyValueTensor(vec, tensorOp)  \
            for(auto i : getTensor(tensorOp)) { vec.push_back(i); }
 
#define replaceValueTensorOp  replaceOpWithNewOp<ValueTensorLiteralOp> 


                   