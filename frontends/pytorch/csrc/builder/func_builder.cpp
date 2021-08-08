//===- func_builder.cpp ---------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "func_builder.h"

#include "op_builder.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "npcomp-c/TorchTypes.h"

using namespace torch_mlir;

std::unique_ptr<FuncBuilder>
FuncBuilder::createFunction(FuncBuilder::Inserter &inserter,
                            MlirLocation location, const std::string &name,
                            std::vector<MlirType> &inputTypes) {
  auto context = mlirLocationGetContext(location);
  // TODO: Create a dedicated API upstream for creating/manipulating func ops.
  // (this is fragile and reveals details that are not guaranteed).
  std::vector<MlirNamedAttribute> funcAttrs;
  funcAttrs.push_back(toMlirNamedAttribute(
      "type", mlirTypeAttrGet(mlirFunctionTypeGet(
                  context, inputTypes.size(), inputTypes.data(),
                  /*numResults=*/0, /*results=*/nullptr))));
  funcAttrs.push_back(toMlirNamedAttribute(
      "sym_name", mlirStringAttrGet(
                      context, mlirStringRefCreate(name.data(), name.size()))));

  MlirOperationState state =
      mlirOperationStateGet(toMlirStringRef("builtin.func"), location);
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

void FuncBuilder::rewriteFuncReturnTypes(std::vector<MlirType> &resultTypes) {
  // Get inputs from current function type.
  MlirAttribute funcTypeAttr =
      mlirOperationGetAttributeByName(funcOp, toMlirStringRef("type"));
  assert(!mlirAttributeIsNull(funcTypeAttr) &&
         "function missing 'type' attribute");
  assert(mlirAttributeIsAType(funcTypeAttr) &&
         "function type is not a TypeAttr");
  MlirType funcType = mlirTypeAttrGetValue(funcTypeAttr);
  std::vector<MlirType> inputTypes;
  for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(funcType); i < e; ++i) {
    inputTypes.push_back(mlirFunctionTypeGetInput(funcType, i));
  }

  // Make new function type.
  MlirType newFuncType =
      mlirFunctionTypeGet(context, inputTypes.size(), inputTypes.data(),
                          resultTypes.size(), resultTypes.data());
  MlirAttribute newFuncTypeAttr = mlirTypeAttrGet(newFuncType);
  mlirOperationSetAttributeByName(funcOp, toMlirStringRef("type"),
                                  newFuncTypeAttr);
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
    MlirType t = npcompTorchIntTypeGet(context);
    MlirAttribute value =
        mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), s.to<int64_t>());
    MlirOperation op = createMlirOperation(
        "torch.constant.int", loc, t, toMlirNamedAttribute("value", value));
    insertConstantOp(op);
    return mlirOperationGetResult(op, 0);
  }
  if (s.isFloatingPoint()) {
    MlirType t = npcompTorchFloatTypeGet(context);
    MlirAttribute value = mlirFloatAttrDoubleGet(
        context, mlirF64TypeGet(context), s.to<double>());
    MlirOperation op = createMlirOperation(
        "torch.constant.float", loc, t, toMlirNamedAttribute("value", value));
    insertConstantOp(op);
    return mlirOperationGetResult(op, 0);
  }
  if (s.isBoolean()) {
    return getBoolConstant(loc, s.to<bool>());
  }
  // TODO: s.isComplex()

  throw std::invalid_argument("TODO: Scalar of unknown kind");
}

MlirValue FuncBuilder::getBoolConstant(MlirLocation loc, bool v) {
  return insertConstantOp(OpBuilder(context).createBoolConstant(loc, v));
}

MlirValue FuncBuilder::getNoneConstant(MlirLocation loc) {
  return insertConstantOp(OpBuilder(context).createNoneConstant(loc));
}

MlirValue FuncBuilder::buildList(MlirLocation loc, MlirType elementType,
                                 std::vector<MlirValue> &elements) {
  auto context = mlirLocationGetContext(loc);
  MlirType resultType = npcompTorchListTypeGet(context, elementType);
  OperationStateHolder state{"torch.prim.ListConstruct", loc};
  mlirOperationStateAddResults(state, 1, &resultType);
  mlirOperationStateAddOperands(state, elements.size(), elements.data());
  MlirOperation op = state.createOperation();
  entryBlock.insertBeforeTerminator(op);
  return mlirOperationGetResult(op, 0);
}
