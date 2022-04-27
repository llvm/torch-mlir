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
#include "torch-mlir-c/TorchTypes.h"

namespace py = pybind11;
using namespace torch_mlir;

using Value = torch::jit::Value;
using Block = torch::jit::Block;
using Node = torch::jit::Node;

namespace {
class NodeImporter {
public:
  NodeImporter(MlirContext context) : context(context) {}

  void importNode(Node *node, MlirBlock appendToBlock);
  MlirBlock importBlock(
      Block *jitBlock, CreateTerminatorFn createTerminator,
      c10::optional<c10::ArrayRef<MlirType>> blockArgTypes = c10::nullopt);

private:
  MlirBlock
  createBlockFor(Block *jitBlock,
                 c10::optional<c10::ArrayRef<MlirType>> blockArgTypes);
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

void NodeImporter::importNode(Node *node, MlirBlock appendToBlock) {
  MlirLocation loc = getMlirLocationFromNode(context, node);
  auto kind = node->kind();

  auto createAndMapTrivialNode = [&](Node *node, const std::string &opName,
                                     InputsTransformFn t) {
    std::vector<MlirValue> mappedInputs = lookupMappedValues(node->inputs());
    MlirOperation operation =
        createMlirOperationAtEnd(appendToBlock, opName, loc,
                                 getMlirTypesFromValues(loc, node->outputs()),
                                 t ? t(mappedInputs) : mappedInputs);
    mapResults(node, operation);
  };

  auto createAndMapNodeWithAttribute = [&](Node *node,
                                           const std::string &opName,
                                           const std::string &attrName,
                                           MlirAttribute attr) {
    MlirOperation operation =
        createMlirOperationAtEnd(appendToBlock, opName, loc,
                                 getMlirTypesFromValues(loc, node->outputs()),
                                 lookupMappedValues(node->inputs()),
                                 toMlirNamedAttribute(attrName.c_str(), attr));
    mapResults(node, operation);
  };

  // Trivial ops with schema.
  auto maybeSchema = node->maybeSchema();
  if (maybeSchema) {
    MlirOperation operation =
        createOperationFromSchema(appendToBlock, loc, node->schema(),
                                  getMlirTypesFromValues(loc, node->outputs()),
                                  lookupMappedValues(node->inputs()));
    mapResults(node, operation);
    return;
  }

  // Builtin interpreter ops with no operator/schema.
  InputsTransformFn transformer =
      kind != c10::prim::DictConstruct ? nullptr : rearrangeDictConstructInputs;
  switch (kind) {
  case c10::prim::ListUnpack:
  case c10::prim::ListConstruct:
  case c10::prim::TupleConstruct:
  case c10::prim::DictConstruct:
  case c10::prim::CreateObject: {
    createAndMapTrivialNode(
        node, "torch.prim." + std::string(kind.toUnqualString()), transformer);
    return;
  }
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
          getMlirTypeFromTorchType(loc, output->type()),
          toMlirNamedAttribute("value",
                               importAttribute(loc, node, c10::attr::value)));
    } else if (output->type()->cast<c10::FloatType>()) {
      op = createMlirOperation(
          "torch.constant.float", loc,
          getMlirTypeFromTorchType(loc, output->type()),
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
          getMlirTypeFromTorchType(loc, output->type()),
          toMlirNamedAttribute(
              "value", mlirStringAttrGet(context, toMlirStringRef(node->s(
                                                      c10::attr::value)))));
    } else if (auto functionType = output->type()->cast<c10::FunctionType>()) {
      torch::jit::Function *function = functionType->function();
      const std::string &symName = function->qualname().qualifiedName();
      op = createMlirOperation(
          "func.constant", loc,
          getFunctionTypeFromSchema(context, function->getSchema()),
          toMlirNamedAttribute(
              "value",
              mlirFlatSymbolRefAttrGet(context, toMlirStringRef(symName))));
    } else if (output->type()->cast<c10::ListType>()) {
      ClassAnnotator dummyAnnotator;
      MlirValue listValue = importIValue(node->ival(c10::attr::value),
                                         appendToBlock,
                                         context,
                                         dummyAnnotator);
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
        getMlirTypesFromValues(loc, node->outputs());
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "torch.prim.Loop", loc, resultTypes,
        lookupMappedValues(node->inputs().slice(0, 2)),
        derefineValues(lookupMappedValues(node->inputs().slice(2)), resultTypes,
                       loc, appendToBlock),
        mlirRegionCreate());
    mapResults(node, operation);
    std::vector<MlirType> terminatorOperandTypes = {
        torchMlirTorchBoolTypeGet(context)};
    terminatorOperandTypes.insert(terminatorOperandTypes.end(),
                                  resultTypes.begin(), resultTypes.end());
    auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                                MlirBlock appendToBlock) {
      createMlirOperationAtEnd(appendToBlock, "torch.prim.Loop.condition", loc,
                               derefineValues(yieldedValues,
                                              terminatorOperandTypes, loc,
                                              appendToBlock));
    };
    mlirRegionAppendOwnedBlock(
        mlirOperationGetRegion(operation, 0),
        importBlock(node->blocks()[0], createTerminator));
    return;
  }

  if (kind == c10::prim::If) {
    std::vector<MlirType> resultTypes =
        getMlirTypesFromValues(loc, node->outputs());
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "torch.prim.If", loc, lookupMappedValue(node->input()),
        resultTypes, mlirRegionCreate(), mlirRegionCreate());
    mapResults(node, operation);
    auto createTerminator = [&](c10::ArrayRef<MlirValue> yieldedValues,
                                MlirBlock appendToBlock) {
      createMlirOperationAtEnd(
          appendToBlock, "torch.prim.If.yield", loc,
          derefineValues(yieldedValues, resultTypes, loc, appendToBlock));
    };
    mlirRegionAppendOwnedBlock(
        mlirOperationGetRegion(operation, 0),
        importBlock(node->blocks()[0], createTerminator));
    mlirRegionAppendOwnedBlock(
        mlirOperationGetRegion(operation, 1),
        importBlock(node->blocks()[1], createTerminator));
    return;
  }

  if (kind == c10::prim::CallMethod) {
    auto classType = node->input(0)->type()->cast<c10::ClassType>();
    auto methodName = node->s(c10::attr::name);
    torch::jit::Function *function = classType->findMethod(methodName);
    MlirType calleeType =
        getFunctionTypeFromSchema(context, function->getSchema());
    std::vector<MlirType> expectedTypes;
    for (int i = 0, e = mlirFunctionTypeGetNumInputs(calleeType); i < e; ++i) {
      expectedTypes.push_back(mlirFunctionTypeGetInput(calleeType, i));
    }
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "torch.prim.CallMethod", loc,
        getMlirTypesFromValues(loc, node->outputs()),
        derefineValues(lookupMappedValues(node->inputs()), expectedTypes, loc,
                       appendToBlock),
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
      return getMlirTypeFromTorchType(loc, v->type());
    });
    MlirOperation operation = createMlirOperationAtEnd(
        appendToBlock, "func.call_indirect", loc,
        getMlirTypesFromValues(loc, node->outputs()),
        lookupMappedValue(node->input(0)),
        derefineValues(lookupMappedValues(node->inputs().slice(1)),
                       expectedTypes, loc, appendToBlock));
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
    c10::optional<c10::ArrayRef<MlirType>> blockArgTypes) {
  MlirBlock block = createBlockFor(jitBlock, blockArgTypes);
  for (Node *node : jitBlock->nodes()) {
    importNode(node, block);
  }
  Node *returnNode = jitBlock->return_node();
  createTerminator(lookupMappedValues(returnNode->inputs()), block);
  return block;
}

static MlirValue adjustBlockArgType(MlirContext context,
                                    MlirBlock appendToBlock, MlirValue value,
                                    MlirType expectedType, MlirLocation loc) {
  MlirType type = mlirValueGetType(value);
  if (mlirTypeEqual(type, expectedType)) {
    return value;
  }
  // For tensors, we might need to erase or add static type information.
  if (torchMlirTypeIsATorchNonValueTensor(type) ||
      torchMlirTypeIsATorchValueTensor(type)) {
    MlirOperation op =
        createMlirOperationAtEnd(appendToBlock, "torch.tensor_static_info_cast",
                                 loc, expectedType, value);
    return mlirOperationGetResult(op, 0);
  }
  {
    std::stringstream msg;
    MlirStringCallback printToStream = +[](MlirStringRef str, void *userData) {
      std::stringstream *stream = static_cast<std::stringstream *>(userData);
      stream->write(str.data, str.length);
    };
    msg << "unhandled: could not adjust formal param type from ";
    mlirTypePrint(type, printToStream, static_cast<void *>(&msg));
    msg << " to expected type ";
    mlirTypePrint(expectedType, printToStream, static_cast<void *>(&msg));
    mlirEmitError(loc, msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
}

MlirBlock NodeImporter::createBlockFor(
    Block *jitBlock, c10::optional<c10::ArrayRef<MlirType>> blockArgTypes) {
  Node *paramNode = jitBlock->param_node();
  MlirLocation loc = getMlirLocationFromNode(context, paramNode);
  std::vector<MlirType> paramNodeTypes =
      getMlirTypesFromValues(loc, paramNode->outputs());
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
    MlirValue adjusted =
        adjustBlockArgType(context, block, value, paramNodeTypes[i], loc);
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
                        c10::optional<c10::ArrayRef<MlirType>> blockArgTypes) {
  NodeImporter importer(context);
  return importer.importBlock(jitBlock, createTerminator, blockArgTypes);
}
