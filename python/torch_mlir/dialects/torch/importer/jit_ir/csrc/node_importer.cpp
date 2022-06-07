//===- node_importer.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "node_importer.h"
#include "torch_to_mlir_utils.h"

#include <unordered_map>

#include "class_annotator.h"
#include "ivalue_importer.h"
#include "mlir_utils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "torch-mlir-c/TorchOps.h"
#include "torch-mlir-c/TorchTypes.h"

using namespace torch_mlir;

using Value = torch::jit::Value;
using Block = torch::jit::Block;
using Node = torch::jit::Node;

namespace {
class NodeImporter {
public:
  NodeImporter(MlirContext context) : context(context) {}

  void importNode(Node *node, MlirBlock appendToBlock,
                  const ImportOptions &importOptions = {});
  MlirBlock importBlock(
      Block *jitBlock, CreateTerminatorFn createTerminator,
      c10::optional<c10::ArrayRef<MlirType>> blockArgTypes = c10::nullopt,
      const ImportOptions &importOptions = {});

private:
  MlirBlock
  createBlockFor(Block *jitBlock,
                 c10::optional<c10::ArrayRef<MlirType>> blockArgTypes,
                 const ImportOptions &importOptions = {});
  void mapValue(Value *jitValue, MlirValue value);
  void mapResults(Node *node, MlirOperation operation);
  MlirValue lookupMappedValue(Value *jitValue);
  std::vector<MlirValue> lookupMappedValues(c10::ArrayRef<Value *> values);

  MlirContext context;
  std::unordered_map<Value *, MlirValue> valueMap;
};
} // namespace

using InputsTransformFn =
    std::function<std::vector<MlirValue>(std::vector<MlirValue> &)>;

// The inputs of `DictConstruct` in TorchScript IR are in the order
// like k0, v0, k1, v1. Rearrange them to put the key operands together and
// then the value operands like k0, k1,v0, v1. This is the expected format by
// the corresponding MLIR op.
static std::vector<MlirValue>
rearrangeDictConstructInputs(std::vector<MlirValue> &inputs) {
  if (inputs.empty())
    return inputs;
  assert(inputs.size() % 2 == 0 &&
         "DictConstruct must have even number of operands");

  std::vector<MlirValue> rearranged;
  std::vector<MlirValue> values;
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    rearranged.push_back(*it);
    values.push_back(*++it);
  }
  rearranged.insert(rearranged.end(), values.begin(), values.end());
  return rearranged;
}

void NodeImporter::importNode(Node *node, MlirBlock appendToBlock,
                              const ImportOptions &importOptions) {
  MlirLocation loc = getMlirLocationFromNode(context, node);
  auto kind = node->kind();

  auto createAndMapTrivialNode = [&](Node *node, const std::string &opName,
                                     InputsTransformFn t) {
    std::vector<MlirValue> mappedInputs = lookupMappedValues(node->inputs());
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, opName, loc,
        getMlirTypesFromValues(loc, node->outputs(), importOptions),
        t ? t(mappedInputs) : mappedInputs);
    mapResults(node, operation);
  };

  auto createAndMapNodeWithAttribute =
      [&](Node *node, const std::string &opName, const std::string &attrName,
          MlirAttribute attr) {
        MlirOperation operation = createMlirOperationAtEnd(
            appendToBlock, opName, loc,
            getMlirTypesFromValues(loc, node->outputs(), importOptions),
            lookupMappedValues(node->inputs()),
            toMlirNamedAttribute(attrName.c_str(), attr));
        mapResults(node, operation);
      };

  // Trivial ops with schema.
  auto maybeSchema = node->maybeSchema();
  if (maybeSchema) {
    MlirOperation operation = createOperationFromSchema(
        appendToBlock, loc, node->schema(),
        getMlirTypesFromValues(loc, node->outputs(), importOptions),
        lookupMappedValues(node->inputs()));
    mapResults(node, operation);
    return;
  }

  // Builtin interpreter ops with no operator/schema.
  switch (kind) {
  case c10::prim::Enter:
  case c10::prim::Exit:
  case c10::prim::ListUnpack:
  case c10::prim::ListConstruct:
  case c10::prim::CreateObject: {
    createAndMapTrivialNode(
        node, "torch.prim." + std::string(kind.toUnqualString()), nullptr);
    return;
  }
  case c10::prim::TupleConstruct: {
    // TODO: We will probably need to adjust the static information for
    // ListConstruct and DictConstruct too.
    auto containedTypes = c10::fmap(
        node->output()->type()->cast<c10::TupleType>()->containedTypes(),
        [&](const c10::TypePtr &t) {
          MlirType type = getMlirTypeFromTorchType(loc, t, importOptions);
          if (mlirTypeIsNull(type)) {
            throw mlir_diagnostic_emitted();
          }
          return type;
        });
    createAndMapTrivialNode(node,
                            "torch.prim." + std::string(kind.toUnqualString()),
                            [&](std::vector<MlirValue> &inputs) {
                              assert(containedTypes.size() == inputs.size());
                              return adjustStaticInformationForValues(
                                  appendToBlock, loc, inputs, containedTypes,
                                  /*userAllowsRefinement=*/true);
                            });
    return;
  }
  case c10::prim::DictConstruct: {
    createAndMapTrivialNode(node,
                            "torch.prim." + std::string(kind.toUnqualString()),
                            rearrangeDictConstructInputs);
    return;
  }
  case c10::prim::Load:
  case c10::prim::Store:
  case c10::prim::GetAttr:
  case c10::prim::SetAttr: {
    createAndMapNodeWithAttribute(
        node, "torch.prim." + std::string(kind.toUnqualString()), "name",
        importAttribute(loc, node, c10::attr::name));
    return;
  }
  }

  if (kind == c10::prim::Constant) {
    auto output = node->output();
    MlirOperation op;
    if (output->type()->cast<c10::NoneType>()) {
      op = createMlirOperation("torch.constant.none", loc,
                               torchMlirTorchNoneTypeGet(context));
    } else if (output->type()->cast<c10::BoolType>()) {
      op = createMlirOperation(
          "torch.constant.bool", loc, torchMlirTorchBoolTypeGet(context),
          toMlirNamedAttribute(
              "value", mlirBoolAttrGet(context, static_cast<bool>(node->i(
                                                    c10::attr::value)))));
    } else if (output->type()->cast<c10::IntType>()) {
      op = createMlirOperation(
          "torch.constant.int", loc,
          getMlirTypeFromTorchType(loc, output->type(), importOptions),
          toMlirNamedAttribute("value",
                               importAttribute(loc, node, c10::attr::value)));
    } else if (output->type()->cast<c10::FloatType>()) {
      op = createMlirOperation(
          "torch.constant.float", loc,
          getMlirTypeFromTorchType(loc, output->type(), importOptions),
          toMlirNamedAttribute("value",
                               importAttribute(loc, node, c10::attr::value)));
    } else if (output->type()->cast<c10::StringType>()) {
      op = createMlirOperation(
          "torch.constant.str", loc, torchMlirTorchStringTypeGet(context),
          toMlirNamedAttribute(
              "value", mlirStringAttrGet(context, toMlirStringRef(node->s(
                                                      c10::attr::value)))));
    } else if (output->type()->cast<c10::TensorType>()) {
      MlirAttribute attr = importAttribute(loc, node, c10::attr::value);
      op = createMlirOperation(
          "torch.tensor.literal", loc,
          torchMlirTorchNonValueTensorTypeGetFromAttribute(attr),
          toMlirNamedAttribute("value", attr));
    } else if (output->type()->cast<c10::DeviceObjType>()) {
      op = createMlirOperation(
          "torch.constant.device", loc,
          getMlirTypeFromTorchType(loc, output->type(), importOptions),
          toMlirNamedAttribute(
              "value", mlirStringAttrGet(context, toMlirStringRef(node->s(
                                                      c10::attr::value)))));
    } else if (auto functionType = output->type()->cast<c10::FunctionType>()) {
      torch::jit::Function *function = functionType->function();
      const std::string &symName = function->qualname().qualifiedName();
      op = createMlirOperation(
          "func.constant", loc,
          getFunctionTypeFromSchema(context, function->getSchema(),
                                    importOptions),
          toMlirNamedAttribute(
              "value",
              mlirFlatSymbolRefAttrGet(context, toMlirStringRef(symName))));
    } else if (output->type()->cast<c10::ListType>()) {
      ClassAnnotator dummyAnnotator;
      MlirValue listValue = importIValue(
          node->ival(c10::attr::value), appendToBlock, context, dummyAnnotator);
      mapResults(node, mlirOpResultGetOwner(listValue));
      return; // Early return, since `importIValue` already added op to block.
    } else {
      std::stringstream msg;
      msg << "unhandled prim::Constant node: ";
      node->print(msg, 0, nullptr);
      mlirEmitError(getMlirLocationFromNode(context, node), msg.str().c_str());
      throw mlir_diagnostic_emitted();
    }
    mlirBlockAppendOwnedOperation(appendToBlock, op);
    mapResults(node, op);
    return;
  }

  if (kind == c10::prim::Loop) {
    std::vector<MlirType> resultTypes =
        getMlirTypesFromValues(loc, node->outputs(), importOptions);
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "torch.prim.Loop", loc, resultTypes,
        lookupMappedValues(node->inputs().slice(0, 2)),
        adjustStaticInformationForValues(
            appendToBlock, loc, lookupMappedValues(node->inputs().slice(2)),
            resultTypes, /*userAllowsRefinement=*/false),
        mlirRegionCreate());
    mapResults(node, operation);
    std::vector<MlirType> terminatorOperandTypes = {
        torchMlirTorchBoolTypeGet(context)};
    terminatorOperandTypes.insert(terminatorOperandTypes.end(),
                                  resultTypes.begin(), resultTypes.end());
    auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                                MlirBlock appendToBlock) {
      createMlirOperationAtEnd(
          appendToBlock, "torch.prim.Loop.condition", loc,
          adjustStaticInformationForValues(appendToBlock, loc, yieldedValues,
                                           terminatorOperandTypes,
                                           /*userAllowsRefinement=*/false));
    };
    mlirRegionAppendOwnedBlock(
        mlirOperationGetRegion(operation, 0),
        importBlock(node->blocks()[0], createTerminator, c10::nullopt, importOptions));
    return;
  }

  if (kind == c10::prim::If) {
    std::vector<MlirType> resultTypes =
        getMlirTypesFromValues(loc, node->outputs(), importOptions);
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "torch.prim.If", loc, lookupMappedValue(node->input()),
        resultTypes, mlirRegionCreate(), mlirRegionCreate());
    mapResults(node, operation);
    auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                                MlirBlock appendToBlock) {
      createMlirOperationAtEnd(
          appendToBlock, "torch.prim.If.yield", loc,
          adjustStaticInformationForValues(appendToBlock, loc, yieldedValues,
                                           resultTypes,
                                           /*userAllowsRefinement=*/false));
    };
    mlirRegionAppendOwnedBlock(
        mlirOperationGetRegion(operation, 0),
        importBlock(node->blocks()[0], createTerminator, c10::nullopt, importOptions));
    mlirRegionAppendOwnedBlock(
        mlirOperationGetRegion(operation, 1),
        importBlock(node->blocks()[1], createTerminator, c10::nullopt, importOptions));
    return;
  }

  if (kind == c10::prim::CallMethod) {
    auto classType = node->input(0)->type()->cast<c10::ClassType>();
    auto methodName = node->s(c10::attr::name);
    torch::jit::Function *function = classType->findMethod(methodName);
    MlirType calleeType =
        getFunctionTypeFromSchema(context, function->getSchema(), importOptions);
    std::vector<MlirType> expectedTypes;
    for (int i = 0, e = mlirFunctionTypeGetNumInputs(calleeType); i < e; ++i) {
      expectedTypes.push_back(mlirFunctionTypeGetInput(calleeType, i));
    }
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "torch.prim.CallMethod", loc,
        getMlirTypesFromValues(loc, node->outputs(), importOptions),
        adjustStaticInformationForValues(
            appendToBlock, loc, lookupMappedValues(node->inputs()),
            expectedTypes, /*userAllowsRefinement=*/false),
        toMlirNamedAttribute("name",
                             importAttribute(loc, node, c10::attr::name)));
    mapResults(node, operation);
    return;
  }

  if (kind == c10::prim::CallFunction) {
    auto functionType = node->input(0)->type()->cast<c10::FunctionType>();
    torch::jit::Block *calleeEntryBlock =
        torch::jit::toGraphFunction(*functionType->function()).graph()->block();
    auto expectedTypes = c10::fmap(calleeEntryBlock->inputs(), [&](Value *v) {
      return getMlirTypeFromTorchType(loc, v->type(), importOptions);
    });
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "func.call_indirect", loc,
        getMlirTypesFromValues(loc, node->outputs(), importOptions),
        lookupMappedValue(node->input(0)),
        adjustStaticInformationForValues(
            appendToBlock, loc, lookupMappedValues(node->inputs().slice(1)),
            expectedTypes, /*userAllowsRefinement=*/false));
    mapResults(node, operation);
    return;
  }

  {
    std::stringstream msg;
    msg << "unhandled: could not import node: ";
    node->print(msg, 0, nullptr);
    mlirEmitError(getMlirLocationFromNode(context, node), msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
}

MlirBlock NodeImporter::importBlock(
    Block *jitBlock, CreateTerminatorFn createTerminator,
    c10::optional<c10::ArrayRef<MlirType>> blockArgTypes,
    const ImportOptions &importOptions) {
  MlirBlock block = createBlockFor(jitBlock, blockArgTypes, importOptions);
  for (Node *node : jitBlock->nodes()) {
    importNode(node, block, importOptions);
  }
  Node *returnNode = jitBlock->return_node();
  createTerminator(lookupMappedValues(returnNode->inputs()), block);
  return block;
}

MlirBlock NodeImporter::createBlockFor(
    Block *jitBlock, c10::optional<c10::ArrayRef<MlirType>> blockArgTypes,
    const ImportOptions &importOptions) {
  Node *paramNode = jitBlock->param_node();
  MlirLocation loc = getMlirLocationFromNode(context, paramNode);
  std::vector<MlirType> paramNodeTypes =
      getMlirTypesFromValues(loc, paramNode->outputs(), importOptions);
  if (!blockArgTypes)
    blockArgTypes = paramNodeTypes;
  else
    assert(blockArgTypes->size() == paramNodeTypes.size());
  std::vector<MlirLocation> blockArgLocs(paramNodeTypes.size(), loc);
  MlirBlock block =
      mlirBlockCreate(blockArgTypes.value().size(),
                      blockArgTypes.value().data(), blockArgLocs.data());
  for (int i = 0, e = mlirBlockGetNumArguments(block); i < e; i++) {
    Value *jitValue = paramNode->outputs()[i];
    MlirValue value = mlirBlockGetArgument(block, i);
    MlirValue adjusted = adjustStaticInformationForValues(
        block, loc, {value}, {paramNodeTypes[i]},
        /*userAllowsRefinement=*/false)[0];
    mapValue(jitValue, adjusted);
  }
  return block;
}

void NodeImporter::mapValue(Value *jitValue, MlirValue value) {
  auto it = valueMap.find(jitValue);
  (void)it;
  assert(it == valueMap.end() && "jitValue has already been mapped");
  valueMap[jitValue] = value;
}
void NodeImporter::mapResults(Node *node, MlirOperation operation) {
  assert(node->outputs().size() ==
         (size_t)mlirOperationGetNumResults(operation));
  for (int i = 0, e = node->outputs().size(); i < e; i++) {
    mapValue(node->outputs()[i], mlirOperationGetResult(operation, i));
  }
}
MlirValue NodeImporter::lookupMappedValue(Value *jitValue) {
  auto it = valueMap.find(jitValue);
  assert(it != valueMap.end() &&
         "trying to get mapping for jitValue that is not mapped yet!");
  return it->second;
}
std::vector<MlirValue>
NodeImporter::lookupMappedValues(c10::ArrayRef<Value *> values) {
  std::vector<MlirValue> ret;
  for (Value *value : values) {
    ret.push_back(lookupMappedValue(value));
  }
  return ret;
}

MlirBlock
torch_mlir::importBlock(MlirContext context, Block *jitBlock,
                        CreateTerminatorFn createTerminator,
                        c10::optional<c10::ArrayRef<MlirType>> blockArgTypes,
                        const ImportOptions &importOptions) {
  NodeImporter importer(context);
  return importer.importBlock(jitBlock, createTerminator, blockArgTypes, importOptions);
}
