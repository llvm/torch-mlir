//===- module_builder.cpp -------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "module_builder.h"
#include "../npcomp_py_interop.h"
#include "graph_importer.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
#include "npcomp-c/Registration.h"

namespace py = pybind11;
using namespace torch_mlir;

static py::object createPythonContextIfNone(py::object contextObj) {
  if (contextObj.is_none()) {
    contextObj = getMlirContextClass()();
  }
  return contextObj;
}

static MlirContext castPythonObjectToMlirContext(py::object &contextObj) {
  assert(!contextObj.is_none() && "context cannot be None");
  auto contextCapsule = contextObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
  MlirContext context = mlirPythonCapsuleToContext(contextCapsule.ptr());
  if (mlirContextIsNull(context)) {
    // An error will have already been set by the above.
    throw py::error_already_set();
  }
  return context;
}

static py::object castMlirModuleToPythonObject(MlirModule module) {
  auto moduleClass = getMlirModuleClass();
  auto moduleCapsule =
      py::reinterpret_steal<py::object>(mlirPythonModuleToCapsule(module));
  return moduleClass.attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(moduleCapsule);
}

static MlirModule createEmptyModule(MlirContext context) {
  // TODO: Extract location from backtrace.
  MlirLocation loc = mlirLocationUnknownGet(context);
  return mlirModuleCreateEmpty(loc);
}

ModuleBuilder::ModuleBuilder(pybind11::object contextObj)
    : contextObj(createPythonContextIfNone(std::move(contextObj))),
      context(castPythonObjectToMlirContext(this->contextObj)),
      module(createEmptyModule(this->context)),
      moduleObj(castMlirModuleToPythonObject(module)),
      unknownLoc(mlirLocationUnknownGet(context)), typeMapper(this->context) {
  // TODO: Rework this once dialect registration C-APIs are in place.
  // https://reviews.llvm.org/D88162
  mlirRegisterAllDialects(context);
  npcompRegisterAllDialects(context);

  // Terminator will always be the first op of an empty module.
  terminator = mlirBlockGetFirstOperation(getBodyBlock());

  // Initialize the python-level MetaModule.
  metaModuleObj = createNpcompMetaModuleClass(moduleObj);
}

std::shared_ptr<AcapController>
ModuleBuilder::startCaptureFunction(std::string name,
                                    std::vector<at::Tensor> args) {
  // TODO: Verify that arguments do not alias each other.
  llvm::SmallVector<MlirType, 4> inputTypes;
  for (auto &arg : args) {
    inputTypes.push_back(typeMapper.forwardTensorToType(arg));
  }

  // TODO: Extract a traceback and use in place of unknownLoc.
  auto inserter = createInserter();
  auto funcBuilder =
      FuncBuilder::createFunction(inserter, unknownLoc, name, inputTypes);
  // Map block arguments.
  MlirBlock entryBlock = funcBuilder->getEntryBlock();
  assert(mlirBlockGetNumArguments(entryBlock) ==
             static_cast<intptr_t>(args.size()) &&
         "entry block incorrect arg arity");
  for (auto it : llvm::enumerate(args)) {
    funcBuilder->mapTensor(it.value(),
                           mlirBlockGetArgument(entryBlock, it.index()));
  }
  auto exportSymbol = py::make_tuple(py::str(name));
  auto exportCallback = [this, name, exportSymbol, inputTypes](
                            llvm::SmallVectorImpl<MlirType> &returnTypes) {
    auto signature =
        GraphMetaFunctionExporter::createSignature(inputTypes, returnTypes);
    auto specializedFunction = createNpcompMetaExportSpecializedFunction(
        /*py ir_symbol_name=*/py::str(name),
        /*py signature=*/std::move(signature));
    getMetaModuleObj().attr("export_symbol")(std::move(exportSymbol),
                                             std::move(specializedFunction));
  };
  return std::make_shared<AcapController>(typeMapper, std::move(funcBuilder),
                                          exportCallback);
}

torch::jit::StrongFunctionPtr
ModuleBuilder::importFunction(torch::jit::StrongFunctionPtr function) {
  auto inserter = createInserter();
  GraphImporter::MlirMappingOptions mappingOptions{
      context,
      llvm::None, // genericFuncName (default to auto)
      llvm::None, // funcName (default to auto)
      typeMapper, inserter,
  };
  auto graphImporter = GraphImporter::forPythonJitFunc(
      function.function_, std::move(mappingOptions));
  graphImporter->initialize();
  graphImporter->importGenericFunc();

  // Export the meta function.
  // Note that when importing "loose" functions like this (not part of a module)
  // we just export it to the top-level as its name in the graph.
  GraphMetaFunctionExporter exporter(metaModuleObj, graphImporter);
  exporter.exportGenericFunction(
      py::make_tuple(py::str(graphImporter->getFuncName())));
  return function;
}

FuncBuilder::Inserter ModuleBuilder::createInserter() {
  MlirBlock block = getBodyBlock();
  MlirOperation terminator = this->terminator;
  return [=](MlirOperation op) {
    mlirBlockInsertOwnedOperationBefore(block, terminator, op);
  };
}

MlirBlock ModuleBuilder::getBodyBlock() {
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  return mlirRegionGetFirstBlock(mlirOperationGetRegion(moduleOp, 0));
}

void ModuleBuilder::bind(py::module &m) {
  py::class_<ModuleBuilder>(m, "ModuleBuilder")
      .def(py::init<py::object>(), py::arg("context") = py::none())
      .def_property_readonly("context", &ModuleBuilder::getContextObj)
      .def_property_readonly("module", &ModuleBuilder::getModuleObj)
      .def_property_readonly("meta_module", &ModuleBuilder::getMetaModuleObj)
      .def("capture_function", &ModuleBuilder::startCaptureFunction,
           py::keep_alive<0, 1>())
      .def("import_function", &ModuleBuilder::importFunction);
}
