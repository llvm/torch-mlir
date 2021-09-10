//===- module_builder.cpp -------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "module_builder.h"

#include "function_importer.h"
#include "ivalue_importer.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Registration.h"
#include "torch-mlir-c/Registration.h"

namespace py = pybind11;
using namespace torch_mlir;

static py::object getMlirIrClass(const char *className) {
  return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr(className);
}

static py::object createPythonContextIfNone(py::object contextObj) {
  if (contextObj.is_none()) {
    contextObj = getMlirIrClass("Context")();
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
  auto moduleClass = getMlirIrClass("Module");
  auto moduleCapsule =
      py::reinterpret_steal<py::object>(mlirPythonModuleToCapsule(module));
  return moduleClass.attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(moduleCapsule);
}

static MlirModule createEmptyModule(MlirContext context) {
  // TODO: Extract location from backtrace.
  MlirLocation loc = mlirLocationUnknownGet(context);
  return mlirModuleCreateEmpty(loc);
}

static std::string
stringifyMlirDiagnosticSeverity(MlirDiagnosticSeverity severity) {
  switch (severity) {
  case MlirDiagnosticError:
    return "error";
  case MlirDiagnosticWarning:
    return "warning";
  case MlirDiagnosticNote:
    return "note";
  case MlirDiagnosticRemark:
    return "remark";
  default:
    return "<unknown severity>";
  }
}

static void printDiagnostic(MlirDiagnostic diagnostic) {
  std::stringstream ss;
  ss << stringifyMlirDiagnosticSeverity(mlirDiagnosticGetSeverity(diagnostic))
     << ": ";
  auto stringCallback = [](MlirStringRef s, void *stringCallbackUserData) {
    auto *ssp = static_cast<std::stringstream *>(stringCallbackUserData);
    ssp->write(s.data, s.length);
  };
  mlirDiagnosticPrint(diagnostic, stringCallback, static_cast<void *>(&ss));
  // Use pybind11's print:
  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html
  py::print(ss.str(),
            py::arg("file") = py::module_::import("sys").attr("stderr"));
}

// Register a diagnostic handler that will redirect output to `sys.stderr`
// instead of a C/C++-level file abstraction. This ensures, for example,
// that mlir diagnostics emitted are correctly routed in Jupyter notebooks.
static void registerPythonSysStderrDiagnosticHandler(MlirContext context) {
  auto diagnosticHandler = [](MlirDiagnostic diagnostic,
                              void *) -> MlirLogicalResult {
    printDiagnostic(diagnostic);
    for (int i = 0, e = mlirDiagnosticGetNumNotes(diagnostic); i != e; i++) {
      printDiagnostic(mlirDiagnosticGetNote(diagnostic, i));
    }
    return mlirLogicalResultSuccess();
  };
  MlirDiagnosticHandlerID id = mlirContextAttachDiagnosticHandler(
      context, diagnosticHandler, nullptr, [](void *) { return; });
  // Ignore the ID. We intend to keep this handler for the entire lifetime
  // of this context.
  (void)id;
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
  torchMlirRegisterAllDialects(context);

  registerPythonSysStderrDiagnosticHandler(context);

  // Terminator will always be the first op of an empty module.
  terminator = mlirBlockGetFirstOperation(getBodyBlock());
}

std::shared_ptr<AcapController>
ModuleBuilder::startCaptureFunction(std::string &name,
                                    std::vector<at::Tensor> args) {
  // TODO: Verify that arguments do not alias each other.
  std::vector<MlirType> inputTypes;
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
  for (size_t i = 0; i < args.size(); ++i) {
    funcBuilder->mapTensor(args[i], mlirBlockGetArgument(entryBlock, i));
  }
  return std::make_shared<AcapController>(typeMapper, std::move(funcBuilder));
}

torch::jit::StrongFunctionPtr
ModuleBuilder::importFunction(torch::jit::StrongFunctionPtr function) {
  MlirBlock block = getBodyBlock();
  MlirOperation terminator = this->terminator;
  MlirOperation func = importJitFunctionAsFuncOp(context, function.function_);
  mlirBlockInsertOwnedOperationBefore(block, terminator, func);
  return function;
}

void ModuleBuilder::importModule(torch::jit::Module jitModule,
                                 py::object maybeClassAnnotator) {
  ClassAnnotator dummyAnnotator;
  ClassAnnotator *classAnnotator = &dummyAnnotator;
  if (!maybeClassAnnotator.is_none()) {
    classAnnotator = py::cast<ClassAnnotator *>(maybeClassAnnotator);
  }
  importIValue(jitModule._ivalue(), mlirModuleGetBody(module),
               mlirModuleGetContext(module), *classAnnotator);
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
      .def("capture_function", &ModuleBuilder::startCaptureFunction,
           py::keep_alive<0, 1>())
      .def("import_function", &ModuleBuilder::importFunction)
      .def("import_module", &ModuleBuilder::importModule, py::arg("module"),
           py::arg("classAnnotator") = py::none());
}
