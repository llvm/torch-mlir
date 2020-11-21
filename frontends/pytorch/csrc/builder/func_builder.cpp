//===- func_builder.cpp ---------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "func_builder.h"

#include "mlir-c/Diagnostics.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
#include "npcomp-c/Types.h"

using namespace torch_mlir;

static MlirOperation createStandardConstant(MlirLocation loc, MlirType type,
                                            MlirAttribute value) {
  OperationStateHolder s("std.constant", loc);
  MlirNamedAttribute valueAttr = mlirNamedAttributeGet("value", value);
  mlirOperationStateAddResults(s, 1, &type);
  mlirOperationStateAddAttributes(s, 1, &valueAttr);
  return s.createOperation();
}

KernelCallBuilder::KernelCallBuilder(MlirContext context, MlirLocation loc,
                                     llvm::StringRef kernelName,
                                     const c10::FunctionSchema &schema)
    : context(context), loc(loc), state("torch.kernel_call", loc),
      kernelName(kernelName), schema(schema) {
  (void)this->context; // Preserve for future.
  MlirNamedAttribute kernelNameAttr = mlirNamedAttributeGet(
      "kernelName",
      mlirStringAttrGet(context, kernelName.size(), kernelName.data()));
  mlirOperationStateAddAttributes(state, 1, &kernelNameAttr);
  addSchemaAttrs();
}

void KernelCallBuilder::addSchemaAttrs() {
  // Map the op schema to the kernel_call attributes:
  //   sigArgTypes
  //   sigRetTypes
  //   sigIsVararg
  //   sigIsVarret
  //   sigIsMutable
  llvm::SmallVector<MlirNamedAttribute, 8> attrs;
  attrs.push_back(mlirNamedAttributeGet(
      "sigIsMutable", mlirBoolAttrGet(context, schema.is_mutable())));
  attrs.push_back(mlirNamedAttributeGet(
      "sigIsVararg", mlirBoolAttrGet(context, schema.is_vararg())));
  attrs.push_back(mlirNamedAttributeGet(
      "sigIsVarret", mlirBoolAttrGet(context, schema.is_varret())));

  // Arg types.
  llvm::SmallVector<MlirAttribute, 4> args;
  for (auto &arg : schema.arguments()) {
    const std::string &typeStr = arg.type()->str();
    args.push_back(mlirStringAttrGet(context, typeStr.size(), typeStr.data()));
  }
  attrs.push_back(mlirNamedAttributeGet(
      "sigArgTypes", mlirArrayAttrGet(context, args.size(), args.data())));

  // Return types.
  llvm::SmallVector<MlirAttribute, 4> returns;
  for (auto &ret : schema.returns()) {
    const std::string &typeStr = ret.type()->str();
    returns.push_back(
        mlirStringAttrGet(context, typeStr.size(), typeStr.data()));
  }
  attrs.push_back(mlirNamedAttributeGet(
      "sigRetTypes",
      mlirArrayAttrGet(context, returns.size(), returns.data())));

  // Add attrs to op.
  mlirOperationStateAddAttributes(state, attrs.size(), attrs.data());
}

void KernelCallBuilder::addOperand(MlirValue operand) {
  mlirOperationStateAddOperands(state, 1, &operand);
}

void KernelCallBuilder::addResultType(MlirType resultType) {
  mlirOperationStateAddResults(state, 1, &resultType);
}

MlirOperation KernelCallBuilder::create() { return state.createOperation(); }

MlirType TypeMapper::mapFromTorchScalarType(c10::ScalarType scalarType) {
  auto type = rawMapFromTorchScalarType(scalarType);
  if (mlirTypeIsNull(type)) {
    std::stringstream message;
    message << "unsupported PyTorch scalar type: " << c10::toString(scalarType);
    throw std::invalid_argument(message.str());
  }
  return type;
}

MlirType TypeMapper::mapFromTorchScalarType(MlirLocation loc,
                                            c10::ScalarType scalarType) {
  auto type = rawMapFromTorchScalarType(scalarType);
  if (mlirTypeIsNull(type)) {
    std::stringstream message;
    message << "unsupported PyTorch scalar type: " << c10::toString(scalarType);
    mlirEmitError(loc, message.str().c_str());
  }
  return type;
}

MlirType TypeMapper::rawMapFromTorchScalarType(c10::ScalarType scalarType) {
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
    return {nullptr};
  }
  }
}

MlirType TypeMapper::mapFromTorchType(MlirLocation loc,
                                      const c10::TypePtr &torchType) {
  using c10::TypeKind;
  auto kind = torchType->kind();
  switch (kind) {
  case TypeKind::TensorType: {
    auto tensorType = torchType->cast<c10::TensorType>();
    // Element type.
    MlirType elementType;
    if (tensorType->scalarType()) {
      elementType = mapFromTorchScalarType(loc, *tensorType->scalarType());
      if (mlirTypeIsNull(elementType))
        return {nullptr};
    } else {
      elementType = npcompAnyDtypeTypeGet(context);
    }
    // Sizes.
    auto &sizes = tensorType->symbolic_sizes();
    if (!sizes.rank()) {
      // Unranked.
      return npcompNdArrayTypeGetUnranked(elementType);
    }
    // Ranked with possibly dynamic dims.
    auto &symbolicShape = tensorType->symbolic_sizes();
    llvm::SmallVector<int64_t, 4> dims;
    dims.resize(*sizes.rank());
    for (size_t i = 0; i < dims.size(); ++i) {
      auto shapeSymbol = symbolicShape[i];
      dims[i] = shapeSymbol.is_static() ? shapeSymbol.static_size() : -1;
    }
    return npcompNdArrayTypeGetRanked(dims.size(), dims.data(), elementType);
  }
  default: {
    std::stringstream message;
    message << "unable to map Torch type " << torchType << " to MLIR type";
    mlirEmitError(loc, message.str().c_str());
    return {nullptr};
  }
  }
}

MlirType TypeMapper::forwardTensorToType(at::Tensor tensor) {
  if (!tensor.defined()) {
    // Undefined tensors are equivalent to None.
    // This may need to be re-evaluated at some point.
    return npcompNoneTypeGet(context);
  }

  MlirType elementType = mapFromTorchScalarType(tensor.scalar_type());
  // TODO: Decide when it is necessary to take strides into account. Right now,
  // just erase them and let the compiler decide.

  auto sizes = tensor.sizes();
  return npcompNdArrayTypeGetRanked(sizes.size(), sizes.data(), elementType);
}

std::unique_ptr<FuncBuilder>
FuncBuilder::createFunction(FuncBuilder::Inserter &inserter,
                            MlirLocation location, llvm::StringRef name,
                            llvm::SmallVectorImpl<MlirType> &inputTypes) {
  auto context = mlirLocationGetContext(location);
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

  inserter(funcOp);
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
    MlirAttribute value = mlirIntegerAttrGet(t, s.to<int64_t>());
    return getGeneralConstant(loc, value);
  }
  if (s.isFloatingPoint()) {
    MlirType t = mlirF64TypeGet(context);
    MlirAttribute value = mlirFloatAttrDoubleGet(context, t, s.to<double>());
    return getGeneralConstant(loc, value);
  }
  if (s.isBoolean()) {
    return getBoolConstant(loc, s.to<bool>());
  }
  // TODO: s.isComplex()

  throw std::invalid_argument("TODO: Scalar of unknown kind");
}

MlirValue FuncBuilder::getBoolConstant(MlirLocation loc, bool v) {
  MlirAttribute value = mlirBoolAttrGet(context, v);
  return getGeneralConstant(loc, value);
}

MlirValue FuncBuilder::getNoneConstant(MlirLocation loc) {
  OperationStateHolder state{"basicpy.singleton", loc};
  MlirType noneType = npcompNoneTypeGet(context);
  mlirOperationStateAddResults(state, 1, &noneType);
  MlirOperation op = state.createOperation();
  return insertConstantOp(op);
}

MlirValue FuncBuilder::getGeneralConstant(MlirLocation loc,
                                          MlirAttribute value) {
  MlirType valueType = mlirAttributeGetType(value);
  MlirOperation constOp = createStandardConstant(loc, valueType, value);
  MlirValue constValue = insertConstantOp(constOp);
  return constValue;
}

MlirValue FuncBuilder::buildList(MlirLocation loc,
                                 llvm::SmallVectorImpl<MlirValue> &elements) {
  MlirType resultType = npcompListTypeGet(context);
  OperationStateHolder state{"basicpy.build_list", loc};
  mlirOperationStateAddResults(state, 1, &resultType);
  mlirOperationStateAddOperands(state, elements.size(), elements.data());
  MlirOperation op = state.createOperation();
  entryBlock.insertBeforeTerminator(op);
  return mlirOperationGetResult(op, 0);
}
