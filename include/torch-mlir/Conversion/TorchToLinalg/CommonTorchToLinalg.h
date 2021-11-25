#ifndef TORCH_COMMON_MLIR_CONVERSION_ATENTOLINALG_ATENTOLINALG_H
#define TORCH_COMMON_MLIR_CONVERSION_ATENTOLINALG_ATENTOLINALG_H

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;


LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                          PatternRewriter &rewriter);

LogicalResult checkNotNone(PatternRewriter &rewriter, Operation *op, Value v);

Value toPositiveDimDynamic(OpBuilder &b, Location loc, Value dim,
                           Value inputRank);

void assertIsValidDim(OpBuilder &b, Location loc, Value dim, Value inputRank);

bool isConstantIntListMatching(Value value, SmallVectorImpl<int64_t> &expects);

Value getDimOp(OpBuilder &b, Location loc, Value v, int dimension);

void checkDimEqualHelper(OpBuilder &b, Location loc, Value lhsDim,
                         Value rhsDim);

SmallVector<Value> getTensorSizesUntilDim(OpBuilder &b, Location loc,
                                          Value tensor, int dim);
Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                           Type elemTy);

Value castIntToIndex(OpBuilder &b, Location loc, Value v);

Value castIndexToInt(OpBuilder &b, Location loc, Value idx);

SmallVector<Value> getTensorSizes(OpBuilder &b, Location loc,
                                         Value tensor);

void populateLoweringMatmulOpPattern(RewritePatternSet &patterns,
                                     TypeConverter &TypeConverter,
                                     MLIRContext *ctx);

#endif // TORCH_COMMON_MLIR_CONVERSION_ATENTOLINALG_ATENTOLINALG_H
