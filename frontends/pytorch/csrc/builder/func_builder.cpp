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
#include "npcomp-c/Types.h"

using namespace torch_mlir;

KernelCallBuilder::KernelCallBuilder(MlirContext context, MlirLocation loc,
                                     const std::string &kernelName,
                                     const c10::FunctionSchema &schema)
    : context(context), loc(loc), state("torch.kernel_call", loc),
      schema(schema) {
  (void)this->context; // Preserve for future.
  MlirNamedAttribute kernelNameAttr = toMlirNamedAttribute(
      "kernelName",
      mlirStringAttrGet(
          context, mlirStringRefCreate(kernelName.data(), kernelName.size())));
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
  std::vector<MlirNamedAttribute> attrs;
  attrs.push_back(toMlirNamedAttribute(
      "sigIsMutable", mlirBoolAttrGet(context, schema.is_mutable())));
  attrs.push_back(toMlirNamedAttribute(
      "sigIsVararg", mlirBoolAttrGet(context, schema.is_vararg())));
  attrs.push_back(toMlirNamedAttribute(
      "sigIsVarret", mlirBoolAttrGet(context, schema.is_varret())));

  // Arg types.
  std::vector<MlirAttribute> args;
  for (auto &arg : schema.arguments()) {
    const std::string &typeStr = arg.type()->str();
    args.push_back(mlirStringAttrGet(
        context, mlirStringRefCreate(typeStr.data(), typeStr.size())));
  }
  attrs.push_back(toMlirNamedAttribute(
      "sigArgTypes", mlirArrayAttrGet(context, args.size(), args.data())));

  // Return types.
  std::vector<MlirAttribute> returns;
  for (auto &ret : schema.returns()) {
    const std::string &typeStr = ret.type()->str();
    returns.push_back(mlirStringAttrGet(
        context, mlirStringRefCreate(typeStr.data(), typeStr.size())));
  }
  attrs.push_back(toMlirNamedAttribute(
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
      mlirOperationStateGet(toMlirStringRef("func"), location);
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
  return insertConstantOp(OpBuilder(context).createBoolConstant(loc, v));
}

MlirValue FuncBuilder::getNoneConstant(MlirLocation loc) {
  return insertConstantOp(OpBuilder(context).createNoneConstant(loc));
}

MlirValue FuncBuilder::getGeneralConstant(MlirLocation loc,
                                          MlirAttribute value) {
  return insertConstantOp(OpBuilder(context).createStdConstant(loc, value));
}

MlirValue FuncBuilder::buildList(MlirLocation loc,
                                 std::vector<MlirValue> &elements) {
  MlirType resultType = npcompListTypeGet(context);
  OperationStateHolder state{"basicpy.build_list", loc};
  mlirOperationStateAddResults(state, 1, &resultType);
  mlirOperationStateAddOperands(state, elements.size(), elements.data());
  MlirOperation op = state.createOperation();
  entryBlock.insertBeforeTerminator(op);
  return mlirOperationGetResult(op, 0);
}
