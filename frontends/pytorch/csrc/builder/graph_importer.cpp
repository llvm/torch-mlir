//===- graph_importer.cpp -------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "graph_importer.h"
#include "../npcomp_py_interop.h"

#include "mlir_utils.h"

#include "mlir-c/Diagnostics.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"

namespace py = pybind11;
using namespace torch_mlir;

//------------------------------------------------------------------------------
// GraphImporter::NodeScope implementation
//------------------------------------------------------------------------------

// A scope of Graph Value * to corresponding MlirValue. Scopes nest
// region-wise. Note that in PyTorch, the thing called 'Block' is analagous
// to a capturing MLIR region.
class GraphImporter::NodeScope {
public:
  NodeScope() = default;
  NodeScope(NodeScope *prev) : prev(prev) {}

  void bindValue(torch::jit::Value *torchValue, MlirValue value);
  MlirValue findValue(torch::jit::Value *torchValue);
  MlirValue findRequiredValue(MlirLocation loc, torch::jit::Value *torchValue);

private:
  llvm::DenseMap<torch::jit::Value *, MlirValue> valueMap;
  NodeScope *prev = nullptr;
};

void GraphImporter::NodeScope::bindValue(torch::jit::Value *torchValue,
                                         MlirValue value) {
  assert(valueMap.count(torchValue) == 0 && "duplicate torch Value bind");
  valueMap[torchValue] = value;
}

MlirValue GraphImporter::NodeScope::findValue(torch::jit::Value *torchValue) {
  auto foundIt = valueMap.find(torchValue);
  if (foundIt == valueMap.end()) {
    if (prev)
      return prev->findValue(torchValue);
    else
      return {nullptr};
  }
  return foundIt->second;
}

MlirValue
GraphImporter::NodeScope::findRequiredValue(MlirLocation loc,
                                            torch::jit::Value *torchValue) {
  MlirValue value = findValue(torchValue);
  if (mlirValueIsNull(value)) {
    std::stringstream msg;
    msg << "internal error: unmapped torch value: %" << torchValue->debugName();
    mlirEmitError(loc, msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
  return value;
}

//------------------------------------------------------------------------------
// GraphImporter::NodeImporter implementation
//------------------------------------------------------------------------------

/// Helper class to import a torch::jit::Node into an MLIR function.
/// This class primarily exists to eliminate the need for large lists of
/// carried arguments related to doing the import.
class GraphImporter::NodeImporter {
public:
  NodeImporter(torch::jit::Node *node, GraphImporter &parent,
               FuncBuilder *funcBuilder, MlirBlock block, MlirOperation ip,
               NodeScope *scope);

  void importNode();
  void importReturnOp();

private:
  MlirContext context() { return parent.context(); }
  void importPrimNode();
  MlirAttribute importValueAttribute();

  torch::jit::Node *node;
  GraphImporter &parent;
  FuncBuilder *funcBuilder;
  MlirBlock block;
  MlirOperation ip;
  NodeScope *scope;
  MlirLocation loc;
};

GraphImporter::NodeImporter::NodeImporter(torch::jit::Node *node,
                                          GraphImporter &parent,
                                          FuncBuilder *funcBuilder,
                                          MlirBlock block, MlirOperation ip,
                                          NodeScope *scope)
    : node(node), parent(parent), funcBuilder(funcBuilder), block(block),
      ip(ip), scope(scope) {
  loc = parent.extractCallstackLoc(node);
}

void GraphImporter::NodeImporter::importNode() {
  // Prim namespace handled specially.
  auto kind = node->kind();
  if (kind.ns() == c10::namespaces::prim) {
    importPrimNode();
    return;
  }

  // Generic import.
  auto funcSchema = node->maybeSchema();
  if (funcSchema) {
    KernelCallBuilder kcb(context(), loc, kind.toQualString(), *funcSchema);
    for (auto *input : node->inputs()) {
      kcb.addOperand(scope->findRequiredValue(loc, input));
    }
    for (auto *output : node->outputs()) {
      MlirType type =
          parent.type_mapper().mapFromTorchType(loc, output->type());
      if (mlirTypeIsNull(type)) {
        throw mlir_diagnostic_emitted();
      }
      kcb.addResultType(type);
    }
    MlirOperation op = kcb.create();
    mlirBlockInsertOwnedOperationBefore(block, ip, op);

    // Map results.
    for (auto it : llvm::enumerate(node->outputs())) {
      scope->bindValue(it.value(), mlirOperationGetResult(op, it.index()));
    }
    return;
  }

  // No soup for you. Not exactly sure when this can happen.
  {
    std::stringstream msg;
    msg << "unhandled: generic operation " << kind.toDisplayString();
    mlirEmitError(loc, msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
}

void GraphImporter::NodeImporter::importReturnOp() {
  OperationStateHolder s("std.return", loc);
  llvm::SmallVector<MlirValue, 4> returnsValues;
  for (auto *input : node->inputs()) {
    returnsValues.push_back(scope->findRequiredValue(loc, input));
  }
  mlirOperationStateAddOperands(s, returnsValues.size(), returnsValues.data());
  mlirBlockInsertOwnedOperationBefore(block, ip, s.createOperation());
}

void GraphImporter::NodeImporter::importPrimNode() {
  auto kind = node->kind();
  if (kind == c10::prim::Constant) {
    auto output = node->output();
    MlirAttribute valueAttr = importValueAttribute();
    MlirValue constValue = funcBuilder->getGeneralConstant(loc, valueAttr);
    scope->bindValue(output, constValue);
    return;
  }

  // Unhandled.
  {
    std::stringstream msg;
    msg << "unhandled: prim operation " << kind.toDisplayString();
    mlirEmitError(loc, msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
}

MlirAttribute GraphImporter::NodeImporter::importValueAttribute() {
  using torch::jit::AttributeKind;
  auto s = c10::attr::value;
  auto kind = node->kindOf(s);
  switch (kind) {
  case AttributeKind::i:
    // TODO: This should be a signed int once we have a constant op that can
    // do that.
    return mlirIntegerAttrGet(mlirIntegerTypeGet(context(), 64), node->i(s));
    break;
  case AttributeKind::f:
    return mlirFloatAttrDoubleGet(context(), mlirF64TypeGet(context()),
                                  node->f(s));
    break;

  default: {
    std::stringstream msg;
    msg << "unhandled: value attribute kind " << toString(kind);
    mlirEmitError(loc, msg.str().c_str());
    throw mlir_diagnostic_emitted();
  }
  }
}

//------------------------------------------------------------------------------
// GraphImporter implementation
//------------------------------------------------------------------------------

GraphImporter::GraphImporter(std::shared_ptr<torch::jit::Graph> graph,
                             MlirMappingOptions mappingOptions)
    : graph(std::move(graph)), mappingOptions(std::move(mappingOptions)) {}

std::shared_ptr<GraphImporter> GraphImporter::forPythonJitFunc(
    torch::jit::Function *function,
    GraphImporter::MlirMappingOptions mappingOptions) {
  // Disallow an attempt to compile a native function.
  if (!function->isGraphFunction()) {
    throw std::invalid_argument(
        "Expected a torch.jit.ScriptFunction with a graph");
  }
  auto graph = function->graph();
  if (!mappingOptions.genericFuncName) {
    mappingOptions.genericFuncName = function->name() + "$generic";
  }
  if (!mappingOptions.funcName) {
    mappingOptions.funcName = function->name();
  }
  return std::make_shared<GraphImporter>(graph, std::move(mappingOptions));
}

void GraphImporter::initialize() {
  defaultLoc = mlirLocationUnknownGet(context());
  // There is not a callstack associated with the graph so, try to grab
  // a location from the first node that has one as a better than nothing
  // thing.
  // TODO: This doesn't actually seem to be working. Investigate when more
  // examples are built out.
  for (auto *node : graph->nodes()) {
    MlirLocation nodeLoc = extractCallstackLoc(node, /*useDefault=*/false);
    if (nodeLoc.ptr) {
      defaultLoc = nodeLoc;
      break;
    }
  }

  // Map inputs.
  MlirLocation inputLoc = extractCallstackLoc(graph->param_node());
  for (const auto &input : graph->inputs()) {
    MlirType t = type_mapper().mapFromTorchType(inputLoc, input->type());
    if (mlirTypeIsNull(t))
      throw mlir_diagnostic_emitted("could not convert function input type");
    genericFuncArgTypes.push_back(t);
  }

  // Map outputs.
  MlirLocation outputLoc = extractCallstackLoc(graph->return_node());
  for (const auto &output : graph->outputs()) {
    MlirType t = type_mapper().mapFromTorchType(outputLoc, output->type());
    if (mlirTypeIsNull(t))
      throw mlir_diagnostic_emitted("could not convert function output type");
    genericFuncReturnTypes.push_back(t);
  }
}

void GraphImporter::importGenericFunc() {
  auto funcBuilder = FuncBuilder::createFunction(
      mappingOptions.inserter, defaultLoc, *mappingOptions.genericFuncName,
      genericFuncArgTypes);
  funcBuilder->rewriteFuncReturnTypes(genericFuncReturnTypes);
  MlirBlock entryBlock = funcBuilder->getEntryBlock();

  // Bind inputs.
  NodeScope scope;
  for (const auto &it : llvm::enumerate(graph->inputs())) {
    MlirValue value = mlirBlockGetArgument(entryBlock, it.index());
    scope.bindValue(it.value(), value);
  }

  // Walk body nodes.
  for (auto *node : graph->nodes()) {
    NodeImporter importer{
        node, *this, funcBuilder.get(), entryBlock, /*ip=*/{nullptr}, &scope};
    importer.importNode();
  }

  // Map the output node to a return.
  auto *outputNode = graph->return_node();
  NodeImporter returnImporter{outputNode,        *this,
                              funcBuilder.get(), entryBlock,
                              /*ip=*/{nullptr},  &scope};
  returnImporter.importReturnOp();
}

MlirLocation GraphImporter::extractCallstackLoc(torch::jit::Node *node,
                                                bool useDefault) {
  auto flc = node->sourceRange().file_line_col();
  if (flc) {
    const std::string &file = std::get<0>(*flc);
    int line = std::get<1>(*flc);
    int col = std::get<2>(*flc);
    return mlirLocationFileLineColGet(context(), toMlirStringRef(file), line,
                                      col);
  }

  return useDefault ? defaultLoc : MlirLocation{nullptr};
}

//------------------------------------------------------------------------------
// GraphMetaExporter implementation.
//------------------------------------------------------------------------------

void GraphMetaFunctionExporter::exportGenericFunction(py::tuple symbolName) {
  auto signature = createSignature(importer().genericFuncArgTypes,
                                   importer().genericFuncReturnTypes);
  auto genericFunction = createNpcompMetaExportGenericFunction(
      /*py ir_symbol_name=*/py::str(*importer().mappingOptions.funcName),
      /*py signature=*/std::move(signature));
  metaModule.attr("export_symbol")(std::move(symbolName),
                                   std::move(genericFunction));
}

py::object GraphMetaFunctionExporter::createSignature(
    const llvm::SmallVectorImpl<MlirType> &argTypes,
    const llvm::SmallVectorImpl<MlirType> &returnTypes) {
  // Effectively:
  //   signature = npcomp.meta.types.Signature(arity)
  auto signature = createNpcompMetaSignatureClass(
      /*py arity=*/argTypes.size());

  // Map arguments.
  // Effectively:
  //   arg_value_types = signature.args
  //   for i, mlir_type in enumerate(importer.genericFuncArgTypes):
  //     value_type = npcomp.types.map_mlir_type_to_meta_type(mlir_type)
  //     arg_value_types[i] = value_type
  py::object argValueTypes = signature.attr("args"); // An npcomp..ValueTypeList
  for (auto it : llvm::enumerate(argTypes)) {
    py::object valueType =
        mapMlirTypeToMetaType(it.value()); // An npcomp..ValueType
    if (!valueType.is_none())
      argValueTypes.attr("__setitem__")(it.index(), std::move(valueType));
  }

  // Map returns.
  // TODO: Signatures always have one result that either becomes None for no
  // result or a tuple for multiple results. Map these. Currently they show
  // as "any".
  if (returnTypes.size() == 1) {
    py::object valueType =
        mapMlirTypeToMetaType(returnTypes[0]); // An npcomp..ValueType
    signature.attr("result") = std::move(valueType);
  }

  return signature;
}
