//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cmath>
#include <cstring>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

int64_t Torch::toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

bool Torch::isValidDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 && dim < inputRank;
}

std::optional<int64_t>
Torch::matchLegalConstantIndexIntoListOfSize(Value v, int64_t length) {
  int64_t dim;
  if (!matchPattern(v, m_TorchConstantInt(&dim)))
    return std::nullopt;
  dim = toPositiveDim(dim, length);
  if (!isValidDim(dim, length))
    return std::nullopt;
  return dim;
}

bool Torch::getListConstructElements(Value v, SmallVectorImpl<Value> &elems) {
  auto listConstruct = v.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return false;
  elems = llvm::to_vector<4>(listConstruct.getElements());
  return true;
}

torch_upstream::ScalarType Torch::getScalarTypeForType(Type type) {
  if (type.isa<Float32Type>())
    return torch_upstream::ScalarType::Float;
  if (type.isa<Float64Type>())
    return torch_upstream::ScalarType::Double;
  if (type.isSignedInteger(64))
    return torch_upstream::ScalarType::Long;
  if (type.isSignedInteger(32))
    return torch_upstream::ScalarType::Int;
  if (type.isSignlessInteger(1))
    return torch_upstream::ScalarType::Bool;
  if (type.isBF16())
    return torch_upstream::ScalarType::BFloat16;
  if (type.isF16())
    return torch_upstream::ScalarType::Half;
  if (type.isUnsignedInteger(8))
    return torch_upstream::ScalarType::Byte;
  if (type.isSignedInteger(8))
    return torch_upstream::ScalarType::Char;
  if (type.isa<ComplexType>()) {
    mlir::Type complexElemType = type.cast<ComplexType>().getElementType();
    if (complexElemType.isF32())
      return torch_upstream::ScalarType::ComplexHalf;
    if (complexElemType.isF64())
      return torch_upstream::ScalarType::ComplexFloat;
    if (complexElemType.isF128())
      return torch_upstream::ScalarType::ComplexDouble;
  }
  llvm::report_fatal_error("unhandled type for getScalarTypeForType");
}

Type Torch::getTypeForTorchType(
    MLIRContext *context, Type type,
    mlir::IntegerType::SignednessSemantics signedness) {
  if (type.isa<Torch::IntType>())
    return IntegerType::get(context, 64, signedness);
  if (type.isa<Torch::FloatType>())
    return Float64Type::get(context);
  llvm::report_fatal_error("unhandled type for getTypeForTorchType");
}

FailureOr<Type>
Torch::getTypeForScalarType(MLIRContext *context,
                            torch_upstream::ScalarType dtypeInt,
                            mlir::IntegerType::SignednessSemantics signedness) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Float:
    return Float32Type::get(context);
  case torch_upstream::ScalarType::Double:
    return Float64Type::get(context);
  case torch_upstream::ScalarType::Long:
    return IntegerType::get(context, 64, signedness);
  case torch_upstream::ScalarType::Int:
    return IntegerType::get(context, 32, signedness);
  case torch_upstream::ScalarType::Bool:
    return IntegerType::get(context, 1);
  case torch_upstream::ScalarType::BFloat16:
    return mlir::FloatType::getBF16(context);
  case torch_upstream::ScalarType::Half:
    return mlir::FloatType::getF16(context);
  case torch_upstream::ScalarType::Byte:
  case torch_upstream::ScalarType::Char:
    return mlir::IntegerType::get(context, 8, signedness);
  case torch_upstream::ScalarType::ComplexHalf:
    return mlir::ComplexType::get(Float32Type::get(context));
  case torch_upstream::ScalarType::ComplexFloat:
    return mlir::ComplexType::get(Float64Type::get(context));
  case torch_upstream::ScalarType::ComplexDouble:
    return mlir::ComplexType::get(Float128Type::get(context));
  case torch_upstream::ScalarType::Undefined:
    return failure();
  default:
    llvm::report_fatal_error("unhandled type for getTypeForScalarType");
  }
}

FailureOr<Type>
Torch::getTorchTypeForScalarType(MLIRContext *context,
                                 torch_upstream::ScalarType dtypeInt) {
  switch (dtypeInt) {
  case torch_upstream::ScalarType::Double:
    return Torch::FloatType::get(context);
  case torch_upstream::ScalarType::Long:
    return Torch::IntType::get(context);
  case torch_upstream::ScalarType::Undefined:
  default:
    return failure();
  }
}

Type Torch::getDefaultDtypeForTorchScalar(Type type) {
  MLIRContext *context = type.getContext();
  if (type.isa<Torch::FloatType>()) {
    // For now, use float32 which is the initial default dtype returned by
    // `torch.get_default_dtype`.
    return Float32Type::get(context);
  }
  if (type.isa<Torch::IntType>())
    return IntegerType::get(context, 64, IntegerType::Signed);
  if (type.isa<Torch::BoolType>())
    return IntegerType::get(context, 1);
  llvm_unreachable(
      "getDefaultDtypeForTorchScalar called on an unsupported type");
}

Type Torch::getBuiltInTypeForTorchScalar(Type type) {
  MLIRContext *context = type.getContext();
  if (type.isa<Torch::FloatType>())
    return Float64Type::get(context);
  if (type.isa<Torch::IntType>())
    return IntegerType::get(context, 64, IntegerType::Signed);
  if (type.isa<Torch::BoolType>())
    return IntegerType::get(context, 1);
  llvm_unreachable(
      "getBuiltInTypeForTorchScalar called on an unsupported type");
}

Value Torch::getDtypeIntValueForType(PatternRewriter &rewriter, Location loc,
                                     Type dtype) {
  int intType = (int)getScalarTypeForType(dtype);
  return rewriter.create<ConstantIntOp>(loc,
                                        rewriter.getI64IntegerAttr(intType));
}

// Helper to convert a tensor to a specific scalar type.
Value Torch::convertTensorToDtype(PatternRewriter &rewriter, Location loc,
                                  Value input, Type dtype) {
  BaseTensorType origType = input.getType().cast<BaseTensorType>();
  Type newType = origType.getWithSizesAndDtype(origType.getSizes(), dtype);
  // `convertIntVal` contains the corresponding integer for the dtype which is
  // used by the aten.to.dtype op.
  Value convertIntVal = getDtypeIntValueForType(rewriter, loc, dtype);
  Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  Value converted = rewriter.create<AtenToDtypeOp>(
      loc, newType, input, convertIntVal, falseVal, falseVal, noneVal);
  return converted;
}

bool Torch::isBuiltInType(Type type) {
  return isa<BuiltinDialect>(type.getDialect());
}

std::optional<unsigned> Torch::getTensorRank(Value tensor) {
  BaseTensorType tensorType = tensor.getType().cast<BaseTensorType>();
  if (!tensorType.hasSizes())
    return std::nullopt;
  return tensorType.getSizes().size();
}

bool Torch::isViewLikeOp(Operation *op) {
  // AtenContiguousOp might return a view, so this is conservatively
  // correct. We could potentially be more precise and identify the cases
  // that it does not return a view and treat those as having value
  // semantics.
  return isa<AtenBroadcastToOp, AtenContiguousOp, AtenDetachOp, AtenExpandAsOp,
             AtenExpandOp, AtenFlattenUsingIntsOp, AtenPermuteOp, AtenReshapeOp,
             Aten_ReshapeAliasOp, AtenSelectIntOp, AtenSliceTensorOp,
             AtenSqueezeDimOp, AtenSqueezeOp, AtenTOp, AtenToDtypeOp,
             AtenTransposeIntOp, AtenUnsqueezeOp, AtenViewOp,
             TensorStaticInfoCastOp, AtenToDtypeLayoutOp, AtenNumpyTOp,
             AtenNarrowOp, AtenToDeviceOp>(op);
}

Value Torch::getConstantWithGivenDtypeAndValue(PatternRewriter &rewriter,
                                               Location loc, float value,
                                               Type dtype) {
  // Creating constants satisfying backend contract.
  if (dtype.isInteger(64) || dtype.isInteger(32) || dtype.isInteger(8) ||
      dtype.isInteger(1))
    return rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr((int64_t)value));
  if (dtype.isF64() || dtype.isF32() || dtype.isF16() || dtype.isBF16())
    return rewriter.create<ConstantFloatOp>(loc,
                                            rewriter.getF64FloatAttr(value));
  llvm::report_fatal_error(
      "unhandled type for getConstantWithGivenDtypeAndValue");
}

// Return the number of elements of a tensor if the shape is static; otherwise,
// return -1.
int64_t Torch::getNumberOfElements(RankedTensorType inputType) {
  if (!inputType.hasStaticShape())
    return -1;
  SmallVector<int64_t> inputShape =
      makeShapeTorchCompatible(inputType.getShape());
  int64_t numel = 1;
  for (int64_t i = 0; i < inputType.getRank(); i++)
    numel *= inputShape[i];
  return numel;
}

SmallVector<int64_t> Torch::makeShapeLLVMCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kUnknownSize);
    if (shape[i] == kUnknownSize)
      updatedShape[i] = kDynamic;
  }
  return updatedShape;
}

SmallVector<int64_t> Torch::makeShapeTorchCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kDynamic);
    if (shape[i] == kDynamic)
      updatedShape[i] = kUnknownSize;
  }
  return updatedShape;
}

using namespace std;

namespace {
// 矩阵乘法
float *mul(float A[], float B[], int N) {
  float *C = new float[N * N]{};
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }

  // 若绝对值小于10^-10,则置为0（这是我自己定的）
  for (int i = 0; i < N * N; i++) {
    if (abs(C[i]) < pow(10, -10)) {
      C[i] = 0;
    }
  }

  return C;
}

// LUP分解
void LUP_Descomposition(float A[], float L[], float U[], int P[], int N) {
  int row = 0;
  for (int i = 0; i < N; i++) {
    P[i] = i;
  }
  for (int i = 0; i < N - 1; i++) {
    float p = 0.0;
    for (int j = i; j < N; j++) {
      if (abs(A[j * N + i]) > p) {
        p = abs(A[j * N + i]);
        row = j;
      }
    }
    if (0 == p) {
      llvm::errs() << "矩阵奇异，无法计算逆\n";
      return;
    }

    // 交换P[i]和P[row]
    int tmp = P[i];
    P[i] = P[row];
    P[row] = tmp;

    float tmp2 = 0.0;
    for (int j = 0; j < N; j++) {
      // 交换A[i][j]和 A[row][j]
      tmp2 = A[i * N + j];
      A[i * N + j] = A[row * N + j];
      A[row * N + j] = tmp2;
    }

    // 以下同LU分解
    float u = A[i * N + i], l = 0.0;
    for (int j = i + 1; j < N; j++) {
      l = A[j * N + i] / u;
      A[j * N + i] = l;
      for (int k = i + 1; k < N; k++) {
        A[j * N + k] = A[j * N + k] - A[i * N + k] * l;
      }
    }
  }

  // 构造L和U
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      if (i != j) {
        L[i * N + j] = A[i * N + j];
      } else {
        L[i * N + j] = 1;
      }
    }
    for (int k = i; k < N; k++) {
      U[i * N + k] = A[i * N + k];
    }
  }
}

// LUP求解方程
float *LUP_Solve(float L[], float U[], int P[], float b[], int N) {
  float *x = new float[N]();
  float *y = new float[N]();

  // 正向替换
  for (int i = 0; i < N; i++) {
    y[i] = b[P[i]];
    for (int j = 0; j < i; j++) {
      y[i] = y[i] - L[i * N + j] * y[j];
    }
  }
  // 反向替换
  for (int i = N - 1; i >= 0; i--) {
    x[i] = y[i];
    for (int j = N - 1; j > i; j--) {
      x[i] = x[i] - U[i * N + j] * x[j];
    }
    x[i] /= U[i * N + i];
  }
  return x;
}

/*****************矩阵原地转置BEGIN********************/

/* 后继 */
int getNext(int i, int m, int n) { return (i % n) * m + i / n; }

/* 前驱 */
int getPre(int i, int m, int n) { return (i % m) * n + i / m; }

/* 处理以下标i为起点的环 */
void movedata(float *mtx, int i, int m, int n) {
  float temp = mtx[i]; // 暂存
  int cur = i;         // 当前下标
  int pre = getPre(cur, m, n);
  while (pre != i) {
    mtx[cur] = mtx[pre];
    cur = pre;
    pre = getPre(cur, m, n);
  }
  mtx[cur] = temp;
}

/* 转置，即循环处理所有环 */
void transpose(float *mtx, int m, int n) {
  for (int i = 0; i < m * n; ++i) {
    int next = getNext(i, m, n);
    while (
        next >
        i) // 若存在后继小于i说明重复,就不进行下去了（只有不重复时进入while循环）
      next = getNext(next, m, n);
    if (next == i) // 处理当前环
      movedata(mtx, i, m, n);
  }
}
} // namespace

// LUP求逆(将每列b求出的各列x进行组装)
float *Torch::LUP_solve_inverse(float A[], int N) {
  // todo: 内存泄漏，先不管
  // 创建矩阵A的副本，注意不能直接用A计算，因为LUP分解算法已将其改变
  float *A_mirror = new float[N * N]();
  float *inv_A = new float[N * N]();  // 最终的逆矩阵（还需要转置）
  float *inv_A_each = new float[N](); // 矩阵逆的各列
  // float *B    =new float[N*N]();
  float *b = new float[N](); // b阵为B阵的列矩阵分量

  for (int i = 0; i < N; i++) {
    float *L = new float[N * N]();
    float *U = new float[N * N]();
    int *P = new int[N]();

    // 构造单位阵的每一列
    for (int i = 0; i < N; i++) {
      b[i] = 0;
    }
    b[i] = 1;

    // 每次都需要重新将A复制一份
    for (int i = 0; i < N * N; i++) {
      A_mirror[i] = A[i];
    }

    LUP_Descomposition(A_mirror, L, U, P, N);

    inv_A_each = LUP_Solve(L, U, P, b, N);
    memcpy(inv_A + i * N, inv_A_each, N * sizeof(float)); // 将各列拼接起来
  }
  transpose(inv_A, N, N); // 由于现在根据每列b算出的x按行存储，因此需转置

  return inv_A;
}

Value Torch::createTensor(IRRewriter &rewriter, Location loc,
                          MLIRContext *context, std::vector<long> shape,
                          std::vector<float> weight) {
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(weight));
  return rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
}

Value Torch::createReshape(IRRewriter &rewriter, Location loc,
                           MLIRContext *context, std::vector<long> shape,
                           Value originVal) {
  // reshape originVal to according shape
  std::vector<Value> values;
  for (auto i : shape) {
    values.push_back(
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i)));
  }
  Value listShape = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange(values));
  Type resultType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                         rewriter.getF32Type());
  // return rewriter.create<AtenReshapeOp>(loc, resultType, originVal,
  // listShape);
  return rewriter.create<AtenViewOp>(loc, resultType, originVal, listShape);
  // todo: figure out why AtenReshapeOp can't lower to lin-on-tensor while
  // AtenViewOp can
}

llvm::SmallPtrSet<Operation *, 16> Torch::getPositiveLayers(Operation *f) {
  // get ops which output is positive
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenReluOp, AtenSigmoidOp>(op)) {
      if (op->getResult(0).getType().isa<ValueTensorType>()) {
        opWorklist.insert(op);
      }
    }
  });
  return opWorklist;
}
