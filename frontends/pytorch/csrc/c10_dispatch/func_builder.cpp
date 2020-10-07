//===- func_builder.cpp ---------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "func_builder.h"

#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
#include "npcomp-c/Types.h"

using namespace torch_mlir;

static MlirOperation createStandardConstant(MlirLocation loc, MlirType type,
                                            MlirAttribute value) {
  OperationStateHolder s("std.constant", loc);
  MlirNamedAttribute valueAttr = mlirNamedAttributeGet("value", value);
  mlirOperationStateAddResults(&s.state, 1, &type);
  mlirOperationStateAddAttributes(&s.state, 1, &valueAttr);
  return s.createOperation();
}

MlirType TypeMapper::mapScalarType(c10::ScalarType scalarType) {
  using c10::ScalarType;
  switch (scalarType) {
  case ScalarType::Byte:
    // TODO: convert to mlirIntegerTypeUnsignedGet once supported.
    return mlirIntegerTypeGet(context, 8);
  case ScalarType::Char:
    return mlirIntegerTypeGet(context, 8);
  case ScalarType::Short:
    // TODO: convert to mlirIntegerTypeSignedGet once supported.
    return mlirIntegerTypeGet(context, 16);
  case ScalarType::Int:
    // TODO: convert to mlirIntegerTypeSignedGet once supported.
    return mlirIntegerTypeGet(context, 32);
  case ScalarType::Long:
    // TODO: convert to mlirIntegerTypeSignedGet once supported.
    return mlirIntegerTypeGet(context, 64);
  case ScalarType::Bool:
    return npcompBoolTypeGet(context);
  case ScalarType::Double:
    return mlirF64TypeGet(context);
  case ScalarType::Float:
    return mlirF32TypeGet(context);
  case ScalarType::BFloat16:
    return mlirBF16TypeGet(context);
  case ScalarType::Half:
    return mlirF16TypeGet(context);
  default: {
    std::stringstream message;
    message << "unsupported PyTorch scalar type: " << c10::toString(scalarType);
    throw std::invalid_argument(message.str());
  }
  }
}

MlirType TypeMapper::forwardTensorToType(at::Tensor tensor) {
  if (!tensor.defined())
    throw std::invalid_argument("Tensor is not defined");

  MlirType elementType = mapScalarType(tensor.scalar_type());
  // TODO: Decide when it is necessary to take strides into account. Right now,
  // just erase them and let the compiler decide.

  auto sizes = tensor.sizes();
  return npcompNdArrayTypeGetRanked(sizes.size(), sizes.data(), elementType);
}

std::unique_ptr<FuncBuilder>
FuncBuilder::createFunction(MlirContext context, MlirLocation location,
                            llvm::StringRef name,
                            llvm::SmallVectorImpl<MlirType> &inputTypes) {
  // TODO: Create a dedicated API upstream for creating/manipulating func ops.
  // (this is fragile and reveals details that are not guaranteed).
  llvm::SmallVector<MlirNamedAttribute, 4> funcAttrs;
  funcAttrs.push_back(mlirNamedAttributeGet(
      "type", mlirTypeAttrGet(mlirFunctionTypeGet(
                  context, inputTypes.size(), inputTypes.data(),
                  /*numResults=*/0, /*results=*/nullptr))));
  funcAttrs.push_back(mlirNamedAttributeGet(
      "sym_name", mlirStringAttrGet(context, name.size(), name.data())));

  MlirOperationState state = mlirOperationStateGet("func", location);
  mlirOperationStateAddAttributes(&state, funcAttrs.size(), funcAttrs.data());
  {
    // Don't access these once ownership transferred.
    MlirRegion newBodyRegion = mlirRegionCreate();
    MlirBlock newEntryBlock =
        mlirBlockCreate(inputTypes.size(), inputTypes.data());
    mlirRegionInsertOwnedBlockAfter(newBodyRegion, {nullptr}, newEntryBlock);
    mlirOperationStateAddOwnedRegions(&state, 1, &newBodyRegion);
  }

  // Need to re-lookup the region/block because we relinquished ownership above.
  MlirOperation funcOp = mlirOperationCreate(&state);
  MlirRegion bodyRegion = mlirOperationGetRegion(funcOp, 0);
  MlirBlock entryBlock = mlirRegionGetFirstBlock(bodyRegion);

  return std::unique_ptr<FuncBuilder>(new FuncBuilder(
      context, funcOp, BlockBuilder(entryBlock, /*returnOp=*/{nullptr}, true)));
}

void FuncBuilder::rewriteFuncReturnTypes(
    llvm::SmallVectorImpl<MlirType> &resultTypes) {
  // Get inputs from current function type.
  MlirAttribute funcTypeAttr = mlirOperationGetAttributeByName(funcOp, "type");
  assert(!mlirAttributeIsNull(funcTypeAttr) &&
         "function missing 'type' attribute");
  assert(mlirAttributeIsAType(funcTypeAttr) &&
         "function type is not a TypeAttr");
  MlirType funcType = mlirTypeAttrGetValue(funcTypeAttr);
  llvm::SmallVector<MlirType, 4> inputTypes;
  for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(funcType); i < e; ++i) {
    inputTypes.push_back(mlirFunctionTypeGetInput(funcType, i));
  }

  // Make new function type.
  MlirType newFuncType =
      mlirFunctionTypeGet(context, inputTypes.size(), inputTypes.data(),
                          resultTypes.size(), resultTypes.data());
  MlirAttribute newFuncTypeAttr = mlirTypeAttrGet(newFuncType);
  mlirOperationSetAttributeByName(funcOp, "type", newFuncTypeAttr);
  (void)newFuncTypeAttr;
}

MlirValue FuncBuilder::insertConstantOp(MlirOperation op) {
  mlirBlockInsertOwnedOperationAfter(entryBlock.getBlock(), prevConstantOp, op);
  prevConstantOp = op;
  return mlirOperationGetResult(op, 0);
}

MlirValue FuncBuilder::lookupTensor(at::Tensor tensor) {
  for (auto it = tensorValueMap.rbegin(), e = tensorValueMap.rend(); it != e;
       ++it) {
    if (it->first.is_same(tensor))
      return it->second;
  }
  return {nullptr};
}

MlirValue FuncBuilder::getScalarConstant(MlirLocation loc, at::Scalar s) {
  // Note that interpreter "scalars" match the Python semantics and are
  // represented as one of double or int64_t, with a special tag for whether
  // it should be interpreted as a bool.
  if (s.isIntegral(/*includeBool=*/false)) {
    // TODO: Switch to a basicpy.constant that works properly with signed
    // integers and then switch this to a signed integer.
    MlirType t = mlirIntegerTypeGet(context, 64);
    MlirOperation op =
        createStandardConstant(loc, t, mlirIntegerAttrGet(t, s.to<int64_t>()));
    return insertConstantOp(op);
  }
  if (s.isFloatingPoint()) {
    MlirType t = mlirF64TypeGet(context);
    MlirOperation op = createStandardConstant(
        loc, t, mlirFloatAttrDoubleGet(context, t, s.to<double>()));
    return insertConstantOp(op);
  }
  // TODO: s.isBoolean()
  // TODO: s.isComplex()

  throw std::invalid_argument("TODO: Scalar of unknown kind");
}
