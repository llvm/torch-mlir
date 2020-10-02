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

MlirType TypeMapper::mapScalarType(c10::ScalarType scalarType) {
  using c10::ScalarType;
  switch (scalarType) {
  case ScalarType::Byte:
    return mlirIntegerTypeUnsignedGet(context, 8);
  case ScalarType::Char:
    return mlirIntegerTypeSignedGet(context, 8);
  case ScalarType::Short:
    return mlirIntegerTypeSignedGet(context, 16);
  case ScalarType::Int:
    return mlirIntegerTypeSignedGet(context, 32);
  case ScalarType::Long:
    return mlirIntegerTypeSignedGet(context, 64);
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

static MlirOperation createEmptyReturnOp(MlirLocation location) {
  MlirOperationState state = mlirOperationStateGet("std.return", location);
  return mlirOperationCreate(&state);
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

  // Create an empty return op (will rework it later as return types become
  // known).
  MlirOperation returnOp = createEmptyReturnOp(location);
  mlirBlockInsertOwnedOperationBefore(entryBlock, {nullptr}, returnOp);

  return std::unique_ptr<FuncBuilder>(new FuncBuilder(
      context, funcOp, BlockBuilder(entryBlock, returnOp, true)));
}

MlirValue FuncBuilder::lookupTensor(at::Tensor tensor) {
  for (auto it = tensorValueMap.rbegin(), e = tensorValueMap.rend(); it != e;
       ++it) {
    if (it->first.is_same(tensor))
      return it->second;
  }
  return {nullptr};
}
