//===- module_builder.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "module_builder.h"

#include "function_importer.h"
#include "ivalue_importer.h"
#include "mlir_utils.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
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
      unknownLoc(mlirLocationUnknownGet(context)) {
  // TODO: Rework this once dialect registration C-APIs are in place.
  // https://reviews.llvm.org/D88162
  torchMlirRegisterAllDialects(context);

  registerPythonSysStderrDiagnosticHandler(context);

  // Terminator will always be the first op of an empty module.
  terminator = mlirBlockGetFirstOperation(getBodyBlock());
}

torch::jit::StrongFunctionPtr
ModuleBuilder::importFunction(torch::jit::StrongFunctionPtr function,
                              py::object maybeImportOptions) {
  ImportOptions importOptions;
  if (!maybeImportOptions.is_none()) {
    importOptions = py::cast<ImportOptions>(maybeImportOptions);
  }
  MlirBlock block = getBodyBlock();
  MlirOperation terminator = this->terminator;
  MlirOperation func = importJitFunctionAsFuncOp(context, function.function_,
        [](int) -> MlirAttribute { return {nullptr}; }, importOptions);
  mlirBlockInsertOwnedOperationBefore(block, terminator, func);
  return function;
}

void ModuleBuilder::importModule(torch::jit::Module jitModule,
                                 py::object maybeClassAnnotator,
                                 py::object maybeImportOptions) {
  ClassAnnotator dummyAnnotator;
  ClassAnnotator *classAnnotator = &dummyAnnotator;
  if (!maybeClassAnnotator.is_none()) {
    classAnnotator = py::cast<ClassAnnotator *>(maybeClassAnnotator);
  }
  ImportOptions importOptions;
  if (!maybeImportOptions.is_none()) {
    importOptions = py::cast<ImportOptions>(maybeImportOptions);
  }

  // Set a debugging name for the MLIR Module based on the jitModule's class
  // name.
  // This is a bit hacky, because we are mutating the enclosing ModuleOp
  // ad-hoc -- this API could be called by users twice and it would overwrite
  // the previous name. But the workflow of calling importModule/importFunction
  // more than once on a ModuleBuilder is fraught with peril (e.g. redundantly
  // importing the compilation unit twice, having multiple root objects
  // with unclear semantic meaning, ...) that for now we will just assume
  // it doesn't happen.
  //
  // We set it to just the last atom of the name because the leading Python
  // package names don't typically contribute much information as far as
  // a simple identifier while debugging (e.g. to name dump files).
  // We have precise information in the module stored via locations and
  // precise `torch.class_type` names.
  //
  // This name is not semantically load-bearing!!!
  auto &name = *jitModule.type()->name();
  auto debugModuleNameAttr = mlirStringAttrGet(
      context, toMlirStringRef(name.atoms()[name.atoms().size() - 1]));
  mlirOperationSetAttributeByName(mlirModuleGetOperation(module),
                                  toMlirStringRef("torch.debug_module_name"),
                                  debugModuleNameAttr);
  importIValue(jitModule._ivalue(), mlirModuleGetBody(module),
               mlirModuleGetContext(module), *classAnnotator, importOptions);
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
      .def("import_function", &ModuleBuilder::importFunction, py::arg("function"),
           py::arg("importOptions") = py::none())
      .def("import_module", &ModuleBuilder::importModule, py::arg("module"),
           py::arg("classAnnotator") = py::none(),
           py::arg("importOptions") = py::none());
}
